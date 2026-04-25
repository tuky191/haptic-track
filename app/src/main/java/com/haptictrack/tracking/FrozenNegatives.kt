package com.haptictrack.tracking

import android.content.Context
import android.util.Log
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Loader for pre-computed background embeddings shipped as a binary asset.
 *
 * These act as a fixed pool of generic-scene "negative examples" used at lock
 * time to:
 *  1. Cold-start the [OnlineClassifier] (so it has a decision boundary from
 *     frame 1, not from the first ~3 confirmed scene negatives).
 *  2. Derive a per-lock adaptive embedding floor from the gallery → negatives
 *     similarity distribution, replacing the static [ReacquisitionEngine.MIN_EMBEDDING_SIMILARITY]
 *     and [ReacquisitionEngine.PERSON_REID_FLOOR].
 *
 * ## Binary format
 *
 * Little-endian, packed:
 * ```
 * | offset | size  | content                             |
 * |  0     | 4     | int32 count                         |
 * |  4     | 4     | int32 dim                           |
 * |  8     | 4*N*D | float32[count][dim] L2-normalized   |
 * ```
 *
 * Each row is L2-normalized so cosine similarity collapses to a dot product
 * at compare time. Producer (Python) is responsible for normalization.
 *
 * Asset names: `frozen_negatives_mnv3.bin` (1280-dim), `frozen_negatives_osnet.bin` (512-dim).
 */
object FrozenNegatives {
    private const val TAG = "FrozenNeg"

    /**
     * Load a frozen-negatives asset by name.
     *
     * Returns an empty list if the asset is missing or malformed — callers
     * fall back to the static floor constants in that case.
     */
    fun load(context: Context, assetName: String, expectedDim: Int): List<FloatArray> {
        return try {
            context.assets.open(assetName).use { stream ->
                val bytes = stream.readBytes()
                if (bytes.size < 8) {
                    Log.w(TAG, "$assetName: file too small (${bytes.size} bytes)")
                    return emptyList()
                }
                val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                val count = buf.int
                val dim = buf.int
                if (count <= 0 || dim != expectedDim) {
                    Log.w(TAG, "$assetName: header mismatch — count=$count dim=$dim expected dim=$expectedDim")
                    return emptyList()
                }
                val expectedBytes = 8 + count * dim * 4
                if (bytes.size != expectedBytes) {
                    Log.w(TAG, "$assetName: size mismatch — got ${bytes.size} expected $expectedBytes")
                    return emptyList()
                }
                val result = ArrayList<FloatArray>(count)
                repeat(count) {
                    val row = FloatArray(dim)
                    buf.asFloatBuffer().get(row)
                    // Advance the underlying buffer past these floats —
                    // asFloatBuffer() doesn't advance the parent.
                    buf.position(buf.position() + dim * 4)
                    result.add(row)
                }
                Log.i(TAG, "$assetName loaded: $count × $dim")
                result
            }
        } catch (e: IOException) {
            Log.w(TAG, "$assetName missing or unreadable: ${e.message}")
            emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "$assetName parse failed: ${e.message}", e)
            emptyList()
        }
    }
}
