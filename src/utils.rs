use crate::error::ColbertError;
use candle_core::Tensor;

/// Normalizes a tensor using L2 normalization along the last dimension.
///
/// A small epsilon is applied to avoid NaNs when a row is entirely zero.
pub fn normalize_l2(v: &Tensor) -> Result<Tensor, ColbertError> {
    let norm_l2 = v
        .sqr()?
        .sum_keepdim(v.rank() - 1)?
        .sqrt()?
        .clamp(1e-12f32, f32::MAX)?;
    v.broadcast_div(&norm_l2).map_err(ColbertError::from)
}
