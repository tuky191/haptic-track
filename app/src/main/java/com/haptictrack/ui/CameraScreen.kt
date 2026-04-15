package com.haptictrack.ui

import android.graphics.RectF
import android.view.MotionEvent
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.input.pointer.pointerInteropFilter
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.rememberMultiplePermissionsState
import com.haptictrack.tracking.TrackedObject
import com.haptictrack.tracking.TrackingStatus
import com.haptictrack.tracking.TrackingUiState
import com.haptictrack.ui.theme.HapticAmber
import com.haptictrack.ui.theme.HapticCyan
import com.haptictrack.ui.theme.HapticGreen
import com.haptictrack.ui.theme.HapticRed
import kotlinx.coroutines.delay

/**
 * Computes the FILL_CENTER transform: scale + offset to map normalized image
 * coordinates (0..1) to screen pixel coordinates.
 */
private data class FillCenterTransform(
    val scale: Float,
    val offsetX: Float,
    val offsetY: Float,
    val mappedWidth: Float,
    val mappedHeight: Float
)

private fun computeFillCenterTransform(
    viewWidth: Float,
    viewHeight: Float,
    imageWidth: Int,
    imageHeight: Int
): FillCenterTransform {
    if (imageWidth <= 0 || imageHeight <= 0) {
        return FillCenterTransform(1f, 0f, 0f, viewWidth, viewHeight)
    }

    val viewAspect = viewWidth / viewHeight
    val imageAspect = imageWidth.toFloat() / imageHeight.toFloat()

    val scale = if (imageAspect > viewAspect) {
        viewHeight / imageHeight
    } else {
        viewWidth / imageWidth
    }

    val mappedWidth = imageWidth * scale
    val mappedHeight = imageHeight * scale
    val offsetX = (viewWidth - mappedWidth) / 2f
    val offsetY = (viewHeight - mappedHeight) / 2f

    return FillCenterTransform(scale, offsetX, offsetY, mappedWidth, mappedHeight)
}

private fun FillCenterTransform.toScreenX(normalizedX: Float): Float =
    offsetX + normalizedX * mappedWidth

private fun FillCenterTransform.toScreenY(normalizedY: Float): Float =
    offsetY + normalizedY * mappedHeight

private fun FillCenterTransform.toNormalizedX(screenX: Float): Float =
    (screenX - offsetX) / mappedWidth

private fun FillCenterTransform.toNormalizedY(screenY: Float): Float =
    (screenY - offsetY) / mappedHeight

@OptIn(ExperimentalPermissionsApi::class, ExperimentalComposeUiApi::class)
@Composable
fun CameraScreen(viewModel: CameraViewModel = viewModel()) {
    val permissions = rememberMultiplePermissionsState(
        listOf(
            android.Manifest.permission.CAMERA,
            android.Manifest.permission.RECORD_AUDIO
        )
    )

    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    if (permissions.allPermissionsGranted) {
        val lifecycleOwner = LocalLifecycleOwner.current
        val context = LocalContext.current
        val previewView = remember {
            PreviewView(context).apply {
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
        }

        LaunchedEffect(Unit) {
            viewModel.startCamera(lifecycleOwner, previewView)
        }

        // Track re-acquire flash: cyan for 500ms after re-acquisition
        var reacquireFlash by remember { mutableStateOf(false) }
        var previousStatus by remember { mutableStateOf(TrackingStatus.IDLE) }

        LaunchedEffect(uiState.status, uiState.trackedObject?.id) {
            if (uiState.status == TrackingStatus.LOCKED &&
                (previousStatus == TrackingStatus.LOST || previousStatus == TrackingStatus.SEARCHING)) {
                reacquireFlash = true
                delay(500)
                reacquireFlash = false
            }
            previousStatus = uiState.status
        }

        Box(modifier = Modifier.fillMaxSize()) {
            // Camera preview
            AndroidView(
                factory = { previewView },
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInteropFilter { event ->
                        if (event.action == MotionEvent.ACTION_DOWN) {
                            val transform = computeFillCenterTransform(
                                previewView.width.toFloat(),
                                previewView.height.toFloat(),
                                uiState.sourceImageWidth,
                                uiState.sourceImageHeight
                            )
                            val normalizedX = transform.toNormalizedX(event.x)
                            val normalizedY = transform.toNormalizedY(event.y)
                            viewModel.onTapToLock(normalizedX, normalizedY)
                            true
                        } else false
                    }
            )

            // Bounding box overlay
            BoundingBoxOverlay(uiState, reacquireFlash)

            // Label overlay (locked object only)
            LockedLabelOverlay(uiState)

            // Status indicator
            StatusBadge(
                state = uiState,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 64.dp)
            )

            // Record button
            RecordButton(
                isRecording = uiState.isRecording,
                onClick = { viewModel.toggleRecording() },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 48.dp)
            )

            // Clear tracking button
            if (uiState.status != TrackingStatus.IDLE) {
                Button(
                    onClick = { viewModel.clearTracking() },
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(top = 64.dp, end = 16.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color.Black.copy(alpha = 0.5f))
                ) {
                    Text("Clear", color = Color.White)
                }
            }
        }
    } else {
        LaunchedEffect(Unit) {
            permissions.launchMultiplePermissionRequest()
        }
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text("Camera and microphone permissions required", color = Color.White)
        }
    }
}

// ---------------------------------------------------------------------------
// Bounding Box Overlay
// ---------------------------------------------------------------------------

@Composable
private fun BoundingBoxOverlay(state: TrackingUiState, reacquireFlash: Boolean) {
    val lockedId = state.trackedObject?.id
    val isLocked = state.status == TrackingStatus.LOCKED
    val isLost = state.status == TrackingStatus.LOST
    val isIdle = state.status == TrackingStatus.IDLE

    // Pulsing alpha for lost/searching state
    val pulseTransition = rememberInfiniteTransition(label = "lostPulse")
    val pulseAlpha by pulseTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 0.8f,
        animationSpec = infiniteRepeatable(
            animation = tween(800, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulseAlpha"
    )

    Canvas(modifier = Modifier.fillMaxSize()) {
        val transform = computeFillCenterTransform(
            size.width, size.height,
            state.sourceImageWidth, state.sourceImageHeight
        )

        if (isIdle) {
            // Idle: show thin corner brackets so user knows what's tappable
            state.detectedObjects.forEach { obj ->
                drawIdleBrackets(obj.boundingBox, transform)
            }
        } else if (isLocked && state.trackedObject != null) {
            // Locked: draw brackets on the single tracked object only
            val color = if (reacquireFlash) HapticCyan else HapticGreen
            drawBrackets(state.trackedObject.boundingBox, transform, color)
        } else if (isLost && state.trackedObject != null) {
            // Lost: pulsing dashed brackets at last known position
            drawDashedBrackets(
                state.trackedObject.boundingBox, transform,
                HapticRed.copy(alpha = pulseAlpha)
            )
        }
    }
}

/** Compute stroke width that scales with box size, clamped to [2, 6] px. */
private fun scaledStroke(boxWidthPx: Float, boxHeightPx: Float): Float {
    return (minOf(boxWidthPx, boxHeightPx) * 0.008f).coerceIn(2f, 6f)
}

/** Map a normalized box to screen pixel coordinates, clamped to canvas bounds. */
private fun DrawScope.mapBox(
    box: RectF,
    transform: FillCenterTransform
): FloatArray {
    return floatArrayOf(
        transform.toScreenX(box.left).coerceIn(0f, size.width),
        transform.toScreenY(box.top).coerceIn(0f, size.height),
        transform.toScreenX(box.right).coerceIn(0f, size.width),
        transform.toScreenY(box.bottom).coerceIn(0f, size.height)
    )
}

/**
 * Draw camera-viewfinder-style corner brackets with shadow.
 */
private fun DrawScope.drawBrackets(
    box: RectF,
    transform: FillCenterTransform,
    color: Color
) {
    val (left, top, right, bottom) = mapBox(box, transform)
    val w = right - left
    val h = bottom - top
    val cornerLen = minOf(w, h) * 0.2f
    val stroke = scaledStroke(w, h)

    // Shadow pass
    val shadow = Color.Black.copy(alpha = 0.4f)
    val shadowOffset = 1.5f
    drawBracketLines(left + shadowOffset, top + shadowOffset, right + shadowOffset, bottom + shadowOffset,
        cornerLen, shadow, stroke)

    // Color pass
    drawBracketLines(left, top, right, bottom, cornerLen, color, stroke)
}

/**
 * Draw dashed corner brackets (for lost/searching state).
 */
private fun DrawScope.drawDashedBrackets(
    box: RectF,
    transform: FillCenterTransform,
    color: Color
) {
    val (left, top, right, bottom) = mapBox(box, transform)
    val w = right - left
    val h = bottom - top
    val cornerLen = minOf(w, h) * 0.2f
    val stroke = scaledStroke(w, h)
    val dashEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 6f), 0f)

    drawBracketLines(left, top, right, bottom, cornerLen, color, stroke, dashEffect)
}

/**
 * Draw the eight bracket line segments (two per corner).
 */
private fun DrawScope.drawBracketLines(
    left: Float, top: Float, right: Float, bottom: Float,
    cornerLen: Float, color: Color, strokeWidth: Float,
    pathEffect: PathEffect? = null
) {
    // Top-left
    drawLine(color, Offset(left, top), Offset(left + cornerLen, top), strokeWidth, pathEffect = pathEffect)
    drawLine(color, Offset(left, top), Offset(left, top + cornerLen), strokeWidth, pathEffect = pathEffect)
    // Top-right
    drawLine(color, Offset(right, top), Offset(right - cornerLen, top), strokeWidth, pathEffect = pathEffect)
    drawLine(color, Offset(right, top), Offset(right, top + cornerLen), strokeWidth, pathEffect = pathEffect)
    // Bottom-left
    drawLine(color, Offset(left, bottom), Offset(left + cornerLen, bottom), strokeWidth, pathEffect = pathEffect)
    drawLine(color, Offset(left, bottom), Offset(left, bottom - cornerLen), strokeWidth, pathEffect = pathEffect)
    // Bottom-right
    drawLine(color, Offset(right, bottom), Offset(right - cornerLen, bottom), strokeWidth, pathEffect = pathEffect)
    drawLine(color, Offset(right, bottom), Offset(right, bottom - cornerLen), strokeWidth, pathEffect = pathEffect)
}

/**
 * Draw thin corner brackets for idle detections — shows what's tappable.
 */
private fun DrawScope.drawIdleBrackets(box: RectF, transform: FillCenterTransform) {
    val (left, top, right, bottom) = mapBox(box, transform)
    val w = right - left
    val h = bottom - top
    val cornerLen = minOf(w, h) * 0.15f
    val stroke = scaledStroke(w, h) * 0.6f
    val color = Color.White.copy(alpha = 0.6f)

    drawBracketLines(left, top, right, bottom, cornerLen, color, stroke)
}

// ---------------------------------------------------------------------------
// Label Overlay — locked object only
// ---------------------------------------------------------------------------

@Composable
private fun LockedLabelOverlay(state: TrackingUiState) {
    if (state.status != TrackingStatus.LOCKED || state.trackedObject == null) return
    val obj = state.trackedObject

    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val parentWidth = constraints.maxWidth.toFloat()
        val parentHeight = constraints.maxHeight.toFloat()
        val density = LocalDensity.current
        val transform = computeFillCenterTransform(
            parentWidth, parentHeight,
            state.sourceImageWidth, state.sourceImageHeight
        )

        val label = obj.label ?: return@BoxWithConstraints

        val left = transform.toScreenX(obj.boundingBox.left)
        val bottom = transform.toScreenY(obj.boundingBox.bottom)
        val boxWidthPx = transform.toScreenX(obj.boundingBox.right) - left
        val boxHeightPx = bottom - transform.toScreenY(obj.boundingBox.top)

        // Scale font with box size, clamped to 10–16sp
        val fontSize = (boxHeightPx * 0.06f).coerceIn(
            with(density) { 10.sp.toPx() },
            with(density) { 16.sp.toPx() }
        )
        val fontSizeSp = with(density) { fontSize.toSp() }

        // Position inside the box, bottom-left with small inset
        val inset = with(density) { 4.dp.toPx() }
        val xPx = left + inset
        val yPx = bottom - fontSize - inset * 2

        Text(
            text = label,
            color = Color.White,
            fontSize = fontSizeSp,
            modifier = Modifier
                .offset(
                    x = with(density) { xPx.toDp() },
                    y = with(density) { yPx.coerceAtLeast(0f).toDp() }
                )
                .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(4.dp))
                .padding(horizontal = 6.dp, vertical = 2.dp)
        )
    }
}

// ---------------------------------------------------------------------------
// Status Badge & Record Button
// ---------------------------------------------------------------------------

@Composable
private fun StatusBadge(state: TrackingUiState, modifier: Modifier = Modifier) {
    val lockedLabel = state.trackedObject?.label

    val (text, color) = when (state.status) {
        TrackingStatus.IDLE -> "Tap an object to track" to Color.White
        TrackingStatus.SEARCHING -> "Searching..." to HapticAmber
        TrackingStatus.LOCKED -> "Tracking: ${lockedLabel ?: "Object"}" to HapticGreen
        TrackingStatus.LOST -> "Lost: ${lockedLabel ?: "Object"} — searching..." to HapticRed
    }

    Text(
        text = text,
        color = color,
        style = MaterialTheme.typography.titleMedium,
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
            .padding(horizontal = 12.dp, vertical = 6.dp)
    )
}

@Composable
private fun RecordButton(isRecording: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Button(
        onClick = onClick,
        modifier = modifier.size(72.dp),
        shape = CircleShape,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isRecording) HapticRed else Color.White
        )
    ) {
        if (isRecording) {
            Canvas(modifier = Modifier.size(24.dp)) {
                drawRect(color = Color.White)
            }
        }
    }
}
