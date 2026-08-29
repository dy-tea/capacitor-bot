use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Context;

use tokio::sync::{mpsc, oneshot};

use capacitor::model::BuildProgress;
use capacitor::model::Model;
use capacitor::recipe::Recipe;
use capacitor::{build, load};

pub type JobId = u64;

/// A submitted training job with handles to its progress stream and result.
pub struct TrainJob {
    pub id: JobId,
    /// Progress updates `(expert_index, total_experts)`.
    pub progress: mpsc::Receiver<(usize, usize)>,
    /// Resolves to the built model (or an error) once training finishes.
    pub result: oneshot::Receiver<anyhow::Result<Arc<Model>>>,
}

struct TrainRequest {
    id: JobId,
    recipe: Recipe,
    /// Where to persist the resulting model binary.
    output_path: PathBuf,
    progress: mpsc::Sender<(usize, usize)>,
    done: oneshot::Sender<anyhow::Result<Arc<Model>>>,
}

/// A bounded, multi-worker training pool.
///
/// A single dispatcher task hands jobs to one of `worker_count` worker tasks,
/// each of which runs one training at a time on a shared rayon pool, so
/// concurrent trainings are bound by the worker count while each build may
/// still use up to that many rayon threads internally.
#[derive(Clone)]
pub struct Trainer {
    tx: mpsc::Sender<TrainRequest>,
    next_id: Arc<AtomicU64>,
}

impl Trainer {
    /// Spawn a training pool with `worker_count` concurrent workers.
    pub fn spawn(worker_count: usize) -> Self {
        let n = worker_count.max(1);

        let (tx, mut rx) = mpsc::channel::<TrainRequest>(n.saturating_mul(2));

        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .thread_name(|i| format!("capacitor-train-{i}"))
                .build()
                .expect("failed to build trainer thread pool"),
        );

        // One inbox per worker (workers own their receiver).
        let mut worker_txs = Vec::with_capacity(n);
        let mut worker_rxs = Vec::with_capacity(n);

        for _ in 0..n {
            let (wtx, wrx) = mpsc::channel::<TrainRequest>(1);
            worker_txs.push(wtx);
            worker_rxs.push(wrx);
        }

        // Dispatcher: round-robin jobs across workers.
        tokio::spawn(async move {
            let mut i = 0usize;

            while let Some(request) = rx.recv().await {
                if worker_txs[i % n].send(request).await.is_err() {
                    i = i.wrapping_add(1);
                }

                i = i.wrapping_add(1);
            }
        });

        // Workers.
        for rx in worker_rxs {
            let pool = Arc::clone(&pool);

            tokio::spawn(async move {
                worker(rx, pool).await;
            });
        }

        Self {
            tx,
            next_id: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Enqueue a model build. Returns a handle to stream progress and await
    /// the resulting model (which is also written to `output_path`).
    pub async fn submit(
        &self,
        _namespace: u64,
        recipe: Recipe,
        output_path: PathBuf,
    ) -> anyhow::Result<TrainJob> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let (progress_tx, progress) = mpsc::channel(16);
        let (done_tx, result) = oneshot::channel();

        self.tx
            .send(TrainRequest {
                id,
                recipe,
                output_path,
                progress: progress_tx,
                done: done_tx,
            })
            .await
            .with_context(|| "training pool is shut down")?;

        Ok(TrainJob {
            id,
            progress,
            result,
        })
    }
}

async fn worker(mut rx: mpsc::Receiver<TrainRequest>, pool: Arc<rayon::ThreadPool>) {
    while let Some(request) = rx.recv().await {
        let TrainRequest {
            id,
            recipe,
            output_path,
            progress,
            done,
        } = request;

        let pool = Arc::clone(&pool);

        let (result_tx, result) = oneshot::channel();

        tokio::task::spawn_blocking(move || {
            let result = pool.install(move || build_to_path(id, recipe, &output_path, &progress));

            let _ = result_tx.send(result);
        })
        .await
        .ok();

        let result = result
            .await
            .unwrap_or_else(|_| Err(anyhow::anyhow!("training task was aborted")));

        let _ = done.send(result);
    }
}

fn build_to_path(
    id: JobId,
    recipe: Recipe,
    output_path: &PathBuf,
    progress: &mpsc::Sender<(usize, usize)>,
) -> anyhow::Result<Arc<Model>> {
    log_progress(id, progress, 0, recipe.experts.num_total);

    let model = build(recipe, |build_progress| match build_progress {
        BuildProgress::BuildExperts { current, total } => {
            log_progress(id, progress, current, total);
        }
        _ => {}
    })?;

    let bytes = model.into_bytes();

    std::fs::write(output_path, &bytes)?;

    let model = load(output_path)?;

    Ok(Arc::new(model))
}

fn log_progress(_id: JobId, tx: &mpsc::Sender<(usize, usize)>, curr: usize, total: usize) {
    if total == 0 {
        return;
    }

    let _ = tx.try_send((curr, total));
}
