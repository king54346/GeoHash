use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::sync::RwLock;

const ALPHABET_SIZE: usize = 32;
// 将十进制数值转为 base32 编码
const BASE32: [char; 32] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];

const DIST_RANK: [i32; 7] = [0, 20, 150, 600, 5000, 20000, 160000];

struct GeoTrieNode<T> {
    children: [Option<Rc<RefCell<GeoTrieNode<T>>>>; ALPHABET_SIZE],
    pass_cnt: u32,
    // how many times this node is passed
    end: bool,
    //geohash 字符串对应的矩形区域
    geoentry: Option<Rc<RefCell<GeoEntry<T>>>>,
}
#[derive(Debug, Clone)]
pub struct GeoEntry<T> {
    // 矩形区域中的点集合，其中 key 通过 ${lat}_${lng} 的格式组成，val 是 point 的引用
    points: Option<HashMap<String, Box<GeoPoint<T>>>>,
    // 矩形区域对应的 geohash 字符串
    hash: String,
}

impl<T: Clone> GeoEntry<T> {
    // 将 point 对应的 key ${lng}_${lat} 转为对应的经纬度
    fn point_key_to_lat_lng(&self, key: &str) -> (f64, f64) {
        let mut iter = key.split("_");
        let lat = iter.next().unwrap().parse::<f64>().unwrap();
        let lng = iter.next().unwrap().parse::<f64>().unwrap();
        (lat, lng)
    }

    fn point_key(&self, lng: f64, lat: f64) -> String {
        format!("{}_{}", lat, lng)
    }
    fn add_point(&mut self, lng: f64, lat: f64, val: T) {
        if self.points.is_none() {
            self.points = Some(HashMap::new());
        }
        let geopoint = GeoPoint {
            lat,
            lng,
            val,
        };
        let point_key = self.point_key(lng, lat);
        self.points.as_mut().unwrap().insert(point_key, Box::new(geopoint));
    }

    //获取到矩形区域内所有的点集合
    fn get_points(&self) -> Vec<GeoPoint<T>> {
        let mut points = Vec::new();
        if let Some(ref points_map) = self.points {
            for (_, point) in points_map.iter() {
                let point1 = GeoPoint {
                    lat: point.lat,
                    lng: point.lng,
                    val: point.val.clone(),
                };
                points.push(point1);
            }
        }
        points
    }

    fn get_hash(&self) -> &String {
        &self.hash
    }
}

#[derive(Debug, Clone)]
pub struct GeoPoint<T> {
    lat: f64,
    lng: f64,
    val: T,
}


impl<T> GeoTrieNode<T> {
    fn new() -> Self {
        GeoTrieNode {
            children: Default::default(),
            pass_cnt: 0,
            end: false,
            geoentry: None,
        }
    }
    fn dfs(&self) -> Vec<Rc<RefCell<GeoEntry<T>>>> {
        let mut entries = Vec::new();
        if self.end {
            if let Some(geoentry) = self.geoentry.as_ref() {
                entries.push(geoentry.clone());
            }
        }
        for child in self.children.iter() {
            if child.is_none() {
                continue;
            }
            let child = child.clone().unwrap();
            let child_borrow = child.borrow();
            if child_borrow.pass_cnt > 0 {
                let mut child_entries = child_borrow.dfs();
                entries.append(&mut child_entries);
            }
        }
        entries
    }
}

pub struct GeoService<T> {
    root: Rc<RefCell<GeoTrieNode<T>>>,
    lock: RwLock<()>,
}

impl<T: Clone> GeoService<T> {
    fn new(root: GeoTrieNode<T>) -> Self {
        GeoService {
            root: Rc::new(RefCell::new(root)),
            lock: RwLock::new(()),
        }
    }
    //Hash：将用户输入的经纬度 lng、lat 转为 geohash 字符串
    // • get：通过传入的 geohash 字符串，获取到对应于矩形区域块的 GEOEntry 实例
    // • add：通过用户传入的经纬度 lng、lat，构造出 point 实例并添加到对应的矩形区域中
    // • get_list_by_prefix：通过用户输入的 geohash 字符串，获取到对应矩形区域块内所有子矩形区域块的 GEOEntry 实例（包含本身）
    // • rem：通过用户输入的 geohash 字符串，删除对应矩形区域块的 GEOEntry
    // • list_by_radius_m：通过用户输入的中心点 lng、lat，以及对应的距离范围 radius，返回范围内所有的点集合
    fn hash(&self, lng: f64, lat: f64) -> String {
        // lng 通过二分转为 20 个二进制 bit 位
        let lngBits = self.get_binary_bits(&mut String::new(), lng, -180.0, 180.0);

        // lat 通过二分转为 20 个二进制 bit 位
        let latBits = self.get_binary_bits(&mut String::new(), lat, -90.0, 90.0);

        // 经纬度交错在一次，每 5 个 bit 一组，转换成 base32 字符串
        let mut hash = String::new();
        let mut five_bits_tmp = String::new();
        for i in 0..40 {
            if i % 2 == 0 {
                five_bits_tmp.push(lngBits.chars().nth(i / 2).unwrap());
            } else {
                five_bits_tmp.push(latBits.chars().nth(i / 2).unwrap());
            }
            if five_bits_tmp.len() == 5 {
                let index = isize::from_str_radix(&five_bits_tmp, 2).unwrap() as usize;
                hash.push(BASE32[index]);
                five_bits_tmp.clear();
            }
        }
        hash
    }
    // 将 base32 编码转为十进制数值
    fn base32_to_index(&self, c: char) -> usize {
        let index = BASE32.iter().position(|&x| x == c).unwrap();
        index
    }
    // 递归二分，直到凑齐指定长度的二进制字符串
    fn get_binary_bits(&self, bits: &mut String, val: f64, mut start: f64, mut end: f64) -> String {
        let mid = (start + end) / 2.0;
        if val < mid {
            bits.push('0');
            end = mid;
        } else {
            bits.push('1');
            start = mid;
        }
        if bits.len() >= 20 {
            return bits.clone();
        }
        self.get_binary_bits(bits, val, start, end)
    }

    fn get_node(&self, hash: &str) -> Option<Rc<RefCell<GeoTrieNode<T>>>> {
        let mut node = self.root.clone();
        for c in hash.chars() {
            let index = self.base32_to_index(c);
            let child = {
                let n = node.borrow();
                if n.children[index].is_none() {
                    return None;
                }
                n.children[index].clone()
            };
            node = child.unwrap();
        }
        Some(node)
    }
    //前缀查询
    fn list_by_prefix(&self, hash: &str) -> Option<Vec<Rc<RefCell<GeoEntry<T>>>>> {
        let node = self.get_node(hash);
        if node.is_none() {
            return None;
        }
        Some(node.unwrap().borrow().dfs())
    }

    // 通过半径 radiusM，获取到矩形区域所需要的 geohash 字符串的位数
    //
    fn get_bits_length_by_radius_m(&self, radius_m: i32) -> Result<i32, &'static str> {
        if radius_m > 160000 || radius_m < 0 {
            return Err("Invalid radius");
        }

        let mut i = 0;
        loop {
            if radius_m <= DIST_RANK[i + 1] {
                return Ok((8 - i) as i32);
            }
            i += 1;
        }
    }

    fn cal_distance(&self, lng1: f64, lat1: f64, lng2: f64, lat2: f64) -> f64 {
        let dx_rad = self.radians(lng1 - lng2);  // 经度差值(弧度)
        let dy_rad = self.radians(lat1 - lat2);  // 纬度差值(弧度)
        let b_rad = self.radians((lat1 + lat2) / 2.0);  // 平均纬度(弧度)

        let lx = dx_rad * 6367000.0 * b_rad.cos();  // 东西距离
        let ly = 6367000.0 * dy_rad;  // 南北距离

        return (lx * lx + ly * ly).sqrt();  // 用平面的矩形对角距离公式计算总距离
    }

    fn radians(&self,degrees: f64) -> f64 {
        degrees * PI / 180.0
    }

    // 根据传入的中心点，获取到周围八个矩形区域内的点集合
    fn get_center_points(&self, lng: f64, lat: f64, radius_m: i32) -> [(f64, f64); 9] {
        let lat_rad = lat.to_radians();
        let dif_lat = (radius_m as f64) / (111132.954 - 559.822 * lat_rad.cos() * 2.0 + 1.175 * lat_rad.cos() * 4.0);
        let dif_lng = (radius_m as f64) / (111412.84 * lat_rad.cos() - 93.5 * lat_rad.cos() * 3.0);

        let adjust = |value: f64, lower: f64, upper: f64, adjust: f64| {
            if value < lower { value + adjust } else if value > upper { value - adjust } else { value }
        };

        let left = adjust(lng - dif_lng, -180.0, 180.0, 360.0);
        let right = adjust(lng + dif_lng, -180.0, 180.0, 360.0);
        let bot = adjust(lat - dif_lat, -90.0, 90.0, 180.0);
        let top = adjust(lat + dif_lat, -90.0, 90.0, 180.0);

        [(left, top), (lng, top), (right, top), (left, lat), (lng, lat), (right, lat), (left, bot), (lng, bot), (right, bot)]
    }


    pub fn get(&self, hash: &str) -> Option<GeoEntry<T>> {
        let _guard = self.lock.read().unwrap();
        let node = self.get_node(hash)?;
        let n = node.borrow();
        let geoentry = n.geoentry.as_ref()?.borrow();
        Some((*geoentry).clone())
    }


    pub fn add(&self, lng: f64, lat: f64, val: T) {
        let hash = self.hash(lng, lat);

        let _guard = self.lock.write().unwrap();

        let node = self.get_node(&hash);
        if let Some(node) = node {
            let mut n = node.borrow_mut();
            if n.end {
                if let Some(geoentry) = n.geoentry.as_mut() {
                    geoentry.borrow_mut().add_point(lng, lat, val);
                }
                return;
            }
        }

        let mut node = self.root.clone();
        for c in hash.chars() {
            let index = self.base32_to_index(c);
            {
                let mut node_borrow = node.borrow_mut();
                if node_borrow.children[index].is_none() {
                    node_borrow.children[index] = Some(Rc::new(RefCell::new(GeoTrieNode::new())));
                }

                if let Some(child) = &node_borrow.children[index] {
                    child.borrow_mut().pass_cnt += 1;
                }
            }
            node = {
                let node_borrow = node.borrow();
                node_borrow.children[index].clone().unwrap()
            };
        }
        let mut n = node.borrow_mut();
        n.end = true;
        n.geoentry = Some(Rc::new(RefCell::new(GeoEntry {
            points: None,
            hash: hash.clone(),
        })));
        let geoentry = n.geoentry.as_mut().unwrap();
        geoentry.borrow_mut().add_point(lng, lat, val);
    }

    //删除
    pub fn rem(&self, hash: &str) -> bool {
        let _guard = self.lock.write().unwrap();
        let node = self.get_node(hash);
        // 如果目标节点为空，或者 end 标识为 false，直接终止流程
        if node.is_none() || !node.unwrap().borrow().end {
            return false;
        }
        // 走到此处，意味着目标节点一定存在，下面开始检索
        // 移动指针从根节点出发
        let mut r = self.root.clone();
        for c in hash.chars() {
            let index = self.base32_to_index(c);
            {
                // 对途径的 child passCnt --
                let mut n = r.borrow_mut();
                let child = n.children[index].clone().unwrap();
                child.borrow_mut().pass_cnt -= 1;
                // 如果某个 child passCnt 减至 0，则直接丢弃整个 child 返回
                if child.borrow().pass_cnt == 0 {
                    n.children[index] = None;
                    return true;
                }
            }// 把 n 的 borrow_mut() 释放掉
            // 移动指针到下一个节点
            r = {
                let node_borrow = r.borrow();
                node_borrow.children[index].clone().unwrap()
            };
        }
        // 走到此处，意味着目标节点一定存在，下面开始删除
        let mut n = r.borrow_mut();
        n.end = false;
        n.geoentry = None;
        true
    }


    pub fn get_list_by_prefix(&self, prefix: &str) -> Vec<GeoEntry<T>> {
        let _guard = self.lock.read().unwrap();
        self.list_by_prefix(prefix)
            .unwrap_or_default()
            .into_iter()
            .map(|node| (*node.borrow()).clone())
            .collect()
    }



    //获取指定范围内的点集合
    // radius 单位为 m.
    // m 转为对应的经纬度 经度1度≈111m；纬度1度≈111m
    // todo 去重，get_bits_length_by_radius_m研究一下
    pub fn list_by_radius_m(&self, lng: f64, lat: f64, radius: i32) -> Vec<GeoPoint<T>> {
        let _guard = self.lock.read().unwrap();
        // 1 根据用户指定的查询范围，确定所需要的 geohash 字符串的长度，保证对应的矩形区域长度大于等于 radiusM
        let i = self.get_bits_length_by_radius_m(2*radius).unwrap_or_else(|err| panic!("{}", err));
        // 2 针对于传入的 lng、lat 中心点，沿着上、下、左、右方向进行偏移，获取到包含自身在内的总共 9 个点的点矩阵
        // 核心是为了保证通过 9 个点获取到的矩形区域一定能完全把检索范围包含在内
        let points = self.get_center_points(lng, lat, radius);
        // 3. 针对这9个点，通过 ListByPrefix 方法，分别取出区域内的所有子 GEOEntry
        let mut entries = Vec::new();
        let mut hashset = HashSet::new();
        for (lng, lat) in points.into_iter() {
            let hash = self.hash(lng, lat)[0..i as usize].to_string();
            if hashset.insert(hash.clone()) {
                let mut entry = self.list_by_prefix(&hash);
                if entry.is_none() {
                    continue;
                }
                entries.append(&mut entry.unwrap());
            }
        }
        // 4. 针对所有 entry，取出其中包含的所有 point
        // 取出 point 之后，计算其与 center point 的相对距离，如果超过范围则进行过滤
        let mut points = Vec::new();
        // 遍历所有的 entry
        for entry in entries.into_iter() {
            // 遍历 entry 中的所有 point
            let entry_borrow = entry.borrow();
            let entry_points = entry_borrow.get_points();
            for point in entry_points.into_iter() {
                // 计算 point 与 center point 的相对距离
                let distance = self.cal_distance(lng, lat, point.lng, point.lat);
                // 如果超过范围则进行过滤
                if distance > radius as f64 {
                    continue;
                }
                points.push(point);
            }
        }
        points
    }

}

#[cfg(test)]
mod tests {
    use std::thread::Thread;
    // GEOHASH：传入一个 point 对应的名称，查看 point 的信息
    // GEOADD：添加一个 point 到 geohash key 当中
    // GEORADIUS：传入一个 point 以及半径，查看指定范围内的点集合
    use super::*;

    #[test]
    fn test() {
        let root = GeoTrieNode::new();
        let geo_service = GeoService::new(root);
        let lng = 101.0;
        let lat = 77.0;
        let val = "北京市";
        geo_service.add(lng, lat, val);
        geo_service.add(120.0, 47.0, "天津市1");
        geo_service.add(12.0, -21.0, "天津市2");
        geo_service.add(120.0, 46.87, "天津市3");
        geo_service.add(178.0, -88.0, "天津市4");
        geo_service.add(116.4074, 39.9042, "北京");
        geo_service.add(121.4737, 31.2304, "上海");

        geo_service.add (113.2644, 23.1291, "广州");
        geo_service.add(114.0579, 22.5431, "深圳");
        geo_service.add (114.3055, 30.5928, "武汉");
        geo_service.add (108.9402, 34.3415, "西安");
        geo_service.add (104.064, 30.6798, "成都");
        geo_service.add(123.4290, 41.7957, "沈阳");
        geo_service.add  (121.6186, 38.9138, "大连");
        geo_service.add (113.6249, 34.7466, "郑州");
        geo_service.add(118.7968, 32.0603, "南京");
        geo_service.add (116.9956, 36.6512, "济南");

        geo_service.add (126.5343, 45.8037, "哈尔滨");
        geo_service.add(120.5852, 31.299, "苏州");
        geo_service.add(120.3119, 31.5691, "无锡");
        geo_service.add (120.1614, 30.293, "杭州");
        geo_service.add (121.5497, 29.8683, "宁波");
        geo_service.add (120.3826, 36.0671, "青岛");
        geo_service.add (125.3235, 43.8163, "长春");
        geo_service.add(112.7526, 37.8735, "太原");
        let hash = geo_service.hash(120.0, 47.0);

        let geoentry = geo_service.get(&hash);
        let mut entry = geoentry.unwrap();
        entry.add_point(120.0, 46.92, "天津市8");
        entry.points.iter().for_each(|point| {
            println!("point: {:?}", point);
        });
        geo_service.get_list_by_prefix("Y").iter().for_each(|entry| {
            println!("entry: {:?}", entry);
        });
        // geo_service.rem(&hash);
        geo_service.list_by_radius_m(120.0, 46.9, 10000).iter().for_each(|point| {
            println!("point: {:?}", point);
        });
        // let x = geoentry.borrow();
        // let vec = x.get_points();
        // println!("vec: {:?}", vec);
        // let geoentry = geoentry.borrow();
        // assert_eq!(geoentry.get_hash(), &hash);
        // let points = geoentry.get_points();
        // assert_eq!(points.len(), 1);
        // let point = points.get(0).unwrap();
        // assert_eq!(point.lat, lat);
        // assert_eq!(point.lng, lng);
        // assert_eq!(point.val, val);
    }
}