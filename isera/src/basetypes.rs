use num_traits::Num;
use num_traits::Signed;
use std::str::FromStr;

pub trait CloneableNum:
    Num
    + PartialOrd
    + FromStr
    + Clone
    + Copy
    + PartialEq
    + std::fmt::Debug
    + num_traits::bounds::Bounded
    + num_traits::cast::ToPrimitive
    + std::ops::AddAssign
    + std::ops::SubAssign
    + Sync
    + Send
    + Sized
    + Signed
{
}

impl CloneableNum for i8 {}
impl CloneableNum for i16 {}
impl CloneableNum for i32 {}
impl CloneableNum for i64 {}
impl CloneableNum for i128 {}
impl CloneableNum for isize {}
impl CloneableNum for f32 {}
impl CloneableNum for f64 {}

#[derive(Debug, Clone)]
pub struct Edges<NUM: CloneableNum> {
    pub source: Vec<usize>,
    pub target: Vec<usize>,
    pub cost: Vec<NUM>,
    pub capacity: Vec<NUM>,
}

#[derive(Debug, Clone)]
pub struct Nodes {
    pub thread: Vec<usize>,
    pub revthread: Vec<usize>,
    pub predecessor: Vec<Option<usize>>,
    pub depth: Vec<usize>,
    pub edge_tree: Vec<usize>,
    pub last: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct GraphState<NUM: CloneableNum> {
    pub potential: Vec<NUM>,
    pub out_base: Vec<usize>,
    pub flow: Vec<NUM>,
    pub state: Vec<NUM>,
}

#[derive(Clone, Debug, Copy, PartialEq, PartialOrd)]
pub struct CustomEdgeIndices<NUM: CloneableNum> {
    pub cost: NUM,
    pub capacity: NUM,
    pub flow: NUM,
}

#[derive(Debug, Clone)]
pub enum Status {
    Optimal,
    DemandGap,
    Iterationlimited,
}

#[derive(Debug, Clone)]
pub struct State<NUM: CloneableNum> {
    pub nodes_state:Nodes,
    pub graph_state:GraphState<NUM>,
    pub edges_state:Edges<NUM>,
    pub status:Status, 
}
