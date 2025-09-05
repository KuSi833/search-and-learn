# Quantised Cascade

1. Generate N beams using 4-bit model
2. Score all beams with PRM
3. For each beam:
   - If PRM score > high_threshold: Keep as-is (4-bit is good enough)
   - If PRM score in middle range: Upgrade to 8-bit and re-execute
   - If PRM score < low_threshold: Prune (not worth pursuing)
4. Continue with mixed-precision beam set
5. Repeat
