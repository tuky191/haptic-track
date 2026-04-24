package com.haptictrack.camera

import android.graphics.Bitmap

/**
 * Fixed-size ring of pre-allocated bitmaps, cycled to avoid per-frame allocations.
 *
 * Allocates [size] bitmaps up front at [width] x [height] ARGB_8888. Callers
 * acquire via [acquire] and return via [release]. If all bitmaps are checked
 * out (consumer slower than producer), [acquire] returns a freshly-allocated
 * bitmap instead of blocking — the extra allocation is the cost of back-pressure,
 * not a crash.
 *
 * Thread-safe: [acquire] and [release] are synchronized on the internal list.
 * Typical hot path has one producer and one consumer so contention is minimal.
 */
class BitmapRing(
    size: Int,
    private val width: Int,
    private val height: Int
) {
    private val free = ArrayDeque<Bitmap>(size)

    init {
        repeat(size) {
            free.addLast(Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888))
        }
    }

    /** Return a bitmap for writing. Never null. */
    fun acquire(): Bitmap = synchronized(free) {
        free.removeFirstOrNull()
    } ?: Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

    /** Return a bitmap to the pool. Drops it if the pool is already full. */
    fun release(bitmap: Bitmap) {
        if (bitmap.isRecycled || bitmap.width != width || bitmap.height != height) {
            if (!bitmap.isRecycled) bitmap.recycle()
            return
        }
        synchronized(free) { free.addLast(bitmap) }
    }

    /** Recycle every pooled bitmap. Call during teardown. */
    fun releaseAll() {
        synchronized(free) {
            free.forEach { if (!it.isRecycled) it.recycle() }
            free.clear()
        }
    }
}
