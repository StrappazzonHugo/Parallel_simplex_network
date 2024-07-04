use crate::basetypes::*;
use num_traits::identities::zero;
use rayon::prelude::*;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct BlockSearch<NUM: CloneableNum> {
    pub phantom: PhantomData<NUM>,
}
#[derive(Debug, Clone, Copy)]
pub struct FirstEligible<NUM: CloneableNum> {
    pub phantom: PhantomData<NUM>,
}
#[derive(Debug, Clone, Copy)]
pub struct BestEligible<NUM: CloneableNum> {
    pub phantom: PhantomData<NUM>,
}
#[derive(Debug, Clone, Copy)]
pub struct ParallelBlockSearch<NUM: CloneableNum> {
    pub phantom: PhantomData<NUM>,
}
#[derive(Debug, Clone, Copy)]
pub struct ParallelBestEligible<NUM: CloneableNum> {
    pub phantom: PhantomData<NUM>,
}

pub trait PivotRules<NUM: CloneableNum> {
    fn find_entering_arc(
        &self,
        edges: &Edges<NUM>,
        graphstate: &GraphState<NUM>,
        index: usize,
        block_size: usize,
    ) -> (Option<usize>, Option<usize>);
}

///////////////////////
///// Pivot rules /////
///////////////////////

//Best Eligible arc
impl<NUM: CloneableNum> PivotRules<NUM> for BestEligible<NUM> {
    fn find_entering_arc(
        &self,
        edges: &Edges<NUM>,
        graphstate: &GraphState<NUM>,
        _index: usize,
        _block_size: usize,
        //return (arc_index, arc_id)
    ) -> (Option<usize>, Option<usize>) {
        let mut min = zero();
        let mut entering_arc = None;
        let mut index = None;
        for i in 0..graphstate.out_base.len() {
            let arc = unsafe { *graphstate.out_base.get_unchecked(i) };
            let rcplus = unsafe {
                *edges.cost.get_unchecked(arc)
                    + *graphstate
                        .potential
                        .get_unchecked(*edges.target.get_unchecked(arc))
            };
            let rcminus = unsafe {
                *graphstate
                    .potential
                    .get_unchecked(*edges.source.get_unchecked(arc))
            };
            let s: NUM = unsafe { *graphstate.state.get_unchecked(arc) };
            if (rcplus < rcminus) ^ (s.is_negative()) {
                let rc = s * (rcplus - rcminus);
                if rc < min {
                    min = rc;
                    entering_arc = Some(arc);
                    index = Some(i);
                }
            } else {
                continue;
            }
        }
        (index, entering_arc)
    }
}

///////////////////////////////
/// SEQUENTIAL BLOCK SEARCH ///
///////////////////////////////

impl<NUM: CloneableNum> PivotRules<NUM> for BlockSearch<NUM> {
    fn find_entering_arc(
        &self,
        edges: &Edges<NUM>,
        graphstate: &GraphState<NUM>,
        mut index: usize,
        block_size: usize,
        //return (arc_index, arc_id)
    ) -> (Option<usize>, Option<usize>) {
        let mut min: NUM = zero();
        let mut entering_arc: Option<usize> = None;
        let mut nb_block_checked = 0;
        //let start = SystemTime::now();
        while nb_block_checked <= (graphstate.out_base.len() / block_size) + 1 {
            nb_block_checked += 1;
            for i in index..(index + std::cmp::min(block_size, graphstate.out_base.len() - index)) {
                let arc = unsafe { *graphstate.out_base.get_unchecked(i) };
                let rcplus = unsafe {
                    *edges.cost.get_unchecked(arc)
                        + *graphstate
                            .potential
                            .get_unchecked(*edges.target.get_unchecked(arc))
                };
                let rcminus = unsafe {
                    *graphstate
                        .potential
                        .get_unchecked(*edges.source.get_unchecked(arc))
                };
                let s: NUM = unsafe { *graphstate.state.get_unchecked(arc) };
                if (rcplus < rcminus) ^ (s.is_negative()) {
                    let rc = s * (rcplus - rcminus);
                    if rc < min {
                        min = rc;
                        entering_arc = Some(arc);
                        index = i;
                    }
                } else {
                    continue;
                }
            }
            if entering_arc.is_some() {
                return (Some(index), entering_arc);
            }
            index = index + block_size;
            if index > graphstate.out_base.len() {
                index = 0;
            }
        }
        (None, None)
    }
}

/////////////////////////////
/// PARALLEL BLOCK SEARCH ///
/////////////////////////////

//parallel iterator inside of the block perf are fine
impl<NUM: CloneableNum> PivotRules<NUM> for ParallelBlockSearch<NUM> {
    fn find_entering_arc(
        &self,
        edges: &Edges<NUM>,
        graphstate: &GraphState<NUM>,
        mut index: usize,
        block_size: usize,
        //return (arc_index, arc_id)
    ) -> (Option<usize>, Option<usize>) {
        let mut entering_arc: Option<(usize, usize, NUM)>;
        let mut nb_block_checked = 0;

        //let start = SystemTime::now();
        while nb_block_checked <= (graphstate.out_base.len() / block_size) + 1 {
            nb_block_checked += 1;

            entering_arc = graphstate.out_base
                [index..(index + std::cmp::min(block_size, graphstate.out_base.len() - index))]
                .par_iter()
                .enumerate()
                .map(|(pos, &arc)| {
                    let rcplus = unsafe {
                        *edges.cost.get_unchecked(arc)
                            + *graphstate
                                .potential
                                .get_unchecked(*edges.target.get_unchecked(arc))
                    };
                    let rcminus = unsafe {
                        *graphstate
                            .potential
                            .get_unchecked(*edges.source.get_unchecked(arc))
                    };

                    let s: NUM = unsafe { *graphstate.state.get_unchecked(arc) };
                    if (rcplus < rcminus) ^ (s.is_negative()) {
                        (pos, arc, (s * (rcplus - rcminus)))
                    } else {
                        (pos, arc, num_traits::one())
                    }
                })
                .min_by(|(_, _, rc1), (_, _, rc2)| (rc1).partial_cmp(&(rc2)).unwrap());
            //.filter(|(_, _, rc)| *rc < zero())

            if entering_arc.is_some() && entering_arc.unwrap().2 < zero() {
                /*match start.elapsed() {
                    Ok(elapsed) => {
                        println!("{:?}", elapsed.as_nanos());
                    }
                    Err(e) => {
                        println!("Error: {e:?}");
                    }
                }*/
                return (
                    Some(index + entering_arc.unwrap().0),
                    Some(entering_arc.unwrap().1),
                );
            }
            index = index + block_size;
            if index > graphstate.out_base.len() {
                index = 0;
            }
        }
        (None, None)
    }
}

fn get_rc_from_arc<NUM: CloneableNum>(
    arc: usize,
    edges: &Edges<NUM>,
    graphstate: &GraphState<NUM>,
) -> NUM {
    let rcplus = unsafe {
        *edges.cost.get_unchecked(arc)
            + *graphstate
                .potential
                .get_unchecked(*edges.target.get_unchecked(arc))
    };
    let rcminus = unsafe {
        *graphstate
            .potential
            .get_unchecked(*edges.source.get_unchecked(arc))
    };
    let s: NUM = unsafe { *graphstate.state.get_unchecked(arc) };
    s * (rcplus - rcminus)
}

//Parallel Best Eligible arc
impl<NUM: CloneableNum> PivotRules<NUM> for ParallelBestEligible<NUM> {
    fn find_entering_arc(
        &self,
        edges: &Edges<NUM>,
        graphstate: &GraphState<NUM>,
        _index: usize,
        _block_size: usize,
        //return (arc_index, arc_id)
    ) -> (Option<usize>, Option<usize>) {
        let (arc, index);
        let candidate = graphstate
            .out_base
            .iter()
            .enumerate()
            .min_by(|(_, &arc1), (_, &arc2)| {
                get_rc_from_arc(arc1, edges, graphstate)
                    .partial_cmp(&get_rc_from_arc(arc2, edges, graphstate))
                    .unwrap()
            });
        if candidate.is_some()
            && get_rc_from_arc(*candidate.unwrap().1, edges, graphstate) < zero()
        {
            index = Some(candidate.unwrap().0);
            arc = Some(*candidate.unwrap().1);
            return (index, arc);
        } else {
            (None, None)
        }
    }
}

/*
fn find_entering_arc(
        &self,
        edges: &Edges<NUM>,
        nodes: &Nodes<NUM>,
        graphstate: &GraphState<NUM>,
        _index: usize,
        _block_size: usize,
        //return (arc_index, arc_id)
    ) -> (Option<usize>, Option<usize>) {
        let thread_nb = rayon::current_num_threads();
        let mut mins = vec![zero(); thread_nb];
        let mut arcs: Vec<(Option<usize>, Option<usize>)> = vec![(None, None); thread_nb];
        let chunk_size: usize = (graphstate.out_base.len() / thread_nb) + 1;
        let chunks: &Vec<&[usize]> = &graphstate.out_base.chunks(chunk_size).collect();
        std::thread::scope(|s| {
            for (i, (rc_cand, candidate)) in std::iter::zip(&mut mins, &mut arcs).enumerate() {
                s.spawn(move || {
                    for (index, &arc) in chunks[i].iter().enumerate() {
                        let rc = graphstate.state[arc]
                            * (edges.cost[arc] - nodes.potential[edges.source[arc]]
                                + nodes.potential[edges.target[arc]]);
                        if rc < *rc_cand {
                            *rc_cand = rc;
                            *candidate = (Some(chunk_size * i + index), Some(arc));
                        }
                    }
                });
            }
        });
        let mut min = mins[0];
        let mut id = 0;
        for (index, rc) in mins.iter().enumerate() {
            if rc < &min {
                min = *rc;
                id = index;
            }
        }

        if min != zero() {
            return arcs[id];
        }
        (None, None)
    }

*/

impl<NUM: CloneableNum> PivotRules<NUM> for FirstEligible<NUM> {
    fn find_entering_arc(
        &self,
        edges: &Edges<NUM>,
        graphstate: &GraphState<NUM>,
        index: usize,
        _block_size: usize,
        //return (arc_index, arc_id)
    ) -> (Option<usize>, Option<usize>) {
        for i in index + 1..graphstate.out_base.len() {
            let arc = graphstate.out_base[i];
            let rc = graphstate.state[arc]
                * (edges.cost[arc] - graphstate.potential[edges.source[arc]]
                    + graphstate.potential[edges.target[arc]]);
            if rc < zero() {
                return (Some(i), Some(arc));
            }
        } //TODO get_unchecked
        for i in 0..index + 1 {
            let arc = unsafe { *graphstate.out_base.get_unchecked(i) };
            let rc = unsafe {
                *graphstate.state.get_unchecked(arc)
                    * (*edges.cost.get_unchecked(arc)
                        - *graphstate
                            .potential
                            .get_unchecked(*edges.source.get_unchecked(arc))
                        + *graphstate
                            .potential
                            .get_unchecked(*edges.target.get_unchecked(arc)))
            };
            if rc < zero() {
                return (Some(i), Some(arc));
            }
        }
        (None, None)
    }
}
