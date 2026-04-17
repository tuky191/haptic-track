package com.haptictrack.ui

import android.graphics.RectF
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Row
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
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asAndroidPath
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.pointerInteropFilter
import kotlin.math.abs
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.haptictrack.tracking.CaptureMode
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

        // --- Animations ---

        // Lock pulse: scale 0.92 → 1.0 on lock/re-acquire
        val lockPulse = remember { Animatable(1f) }
        var previousStatus by remember { mutableStateOf(TrackingStatus.IDLE) }
        var previousLockedId by remember { mutableStateOf<Int?>(null) }

        // Re-acquire flash color blend (0 = green, 1 = cyan)
        var reacquireBrightness by remember { mutableStateOf(0f) }
        val reacquireColorBlend by animateFloatAsState(
            targetValue = reacquireBrightness,
            animationSpec = tween(300, easing = FastOutSlowInEasing),
            label = "reacquireColor"
        )

        // Lost fade-out: opacity decays from 1.0 → 0.0 over 2s
        val lostOpacity by animateFloatAsState(
            targetValue = if (uiState.status == TrackingStatus.LOST) 0f else 1f,
            animationSpec = tween(
                durationMillis = if (uiState.status == TrackingStatus.LOST) 2000 else 0,
                easing = LinearEasing
            ),
            label = "lostFade"
        )

        // Bracket opacity: smooth transition between states
        val bracketOpacity by animateFloatAsState(
            targetValue = when (uiState.status) {
                TrackingStatus.LOCKED -> 1f
                TrackingStatus.LOST -> lostOpacity
                TrackingStatus.IDLE -> 1f
                TrackingStatus.SEARCHING -> 0f
            },
            animationSpec = tween(200, easing = FastOutSlowInEasing),
            label = "bracketOpacity"
        )

        LaunchedEffect(uiState.status, uiState.trackedObject?.id) {
            val currentId = uiState.trackedObject?.id
            val justLocked = uiState.status == TrackingStatus.LOCKED && previousStatus == TrackingStatus.IDLE
            val justReacquired = uiState.status == TrackingStatus.LOCKED &&
                (previousStatus == TrackingStatus.LOST || previousStatus == TrackingStatus.SEARCHING)
            val idChanged = currentId != null && currentId != previousLockedId && uiState.status == TrackingStatus.LOCKED

            if (justLocked || justReacquired || idChanged) {
                // Lock pulse: scale down then back
                lockPulse.snapTo(0.92f)
                lockPulse.animateTo(1f, tween(200, easing = FastOutSlowInEasing))
            }

            if (justReacquired) {
                // Cyan flash → green
                reacquireBrightness = 1f
                delay(300)
                reacquireBrightness = 0f
            }

            previousStatus = uiState.status
            previousLockedId = currentId
        }

        // --- Zoom indicator fade-out ---
        val zoomIndicatorOpacity = remember { Animatable(0f) }
        LaunchedEffect(uiState.showZoomIndicator) {
            if (uiState.showZoomIndicator) {
                zoomIndicatorOpacity.snapTo(1f)
            } else {
                // Fade out over 500ms
                zoomIndicatorOpacity.animateTo(0f, tween(500, easing = FastOutSlowInEasing))
            }
        }

        // Pinch-to-zoom detector — lives outside recomposition
        val scaleDetector = remember {
            ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    viewModel.onPinchZoom(detector.scaleFactor)
                    return true
                }
                override fun onScaleEnd(detector: ScaleGestureDetector) {
                    // Start the fade-out after a short hold
                    viewModel.hideZoomIndicator()
                }
            })
        }
        // Track whether a scale gesture is in progress to suppress tap-on-release
        var isScaling by remember { mutableStateOf(false) }

        Box(modifier = Modifier.fillMaxSize()) {
            // Camera preview
            AndroidView(
                factory = { previewView },
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInteropFilter { event ->
                        scaleDetector.onTouchEvent(event)

                        if (scaleDetector.isInProgress) {
                            isScaling = true
                            return@pointerInteropFilter true
                        }

                        when (event.action) {
                            MotionEvent.ACTION_DOWN -> {
                                isScaling = false
                                true
                            }
                            MotionEvent.ACTION_UP -> {
                                if (!isScaling && event.pointerCount <= 1) {
                                    val transform = computeFillCenterTransform(
                                        previewView.width.toFloat(),
                                        previewView.height.toFloat(),
                                        uiState.sourceImageWidth,
                                        uiState.sourceImageHeight
                                    )
                                    val normalizedX = transform.toNormalizedX(event.x)
                                    val normalizedY = transform.toNormalizedY(event.y)
                                    viewModel.onTapToLock(normalizedX, normalizedY)
                                }
                                isScaling = false
                                true
                            }
                            else -> true
                        }
                    }
            )

            // Bounding box / contour overlay
            TrackingOverlay(
                state = uiState,
                bracketOpacity = bracketOpacity,
                lockScale = lockPulse.value,
                reacquireColorBlend = reacquireColorBlend,
                lostOpacity = lostOpacity
            )

            // Label overlay (locked object only — kept for testing, remove later)
            LockedLabelOverlay(uiState)

            // Status indicator
            StatusBadge(
                state = uiState,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 64.dp)
            )

            // Zoom level indicator — always visible when tracking, fades after pinch otherwise
            val showZoomAlways = uiState.status == TrackingStatus.LOCKED || uiState.status == TrackingStatus.LOST
            val zoomAlpha = if (showZoomAlways) 1f else zoomIndicatorOpacity.value
            if (zoomAlpha > 0.01f) {
                Text(
                    text = "×${"%.1f".format(uiState.currentZoomRatio)}",
                    color = Color.White.copy(alpha = zoomAlpha),
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(top = 100.dp)
                        .background(
                            Color.Black.copy(alpha = 0.5f * zoomAlpha),
                            RoundedCornerShape(16.dp)
                        )
                        .padding(horizontal = 12.dp, vertical = 4.dp)
                )
            }

            // Capture mode selector (swipeable)
            CaptureModePill(
                captureMode = uiState.captureMode,
                onToggle = { viewModel.toggleCaptureMode() },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 132.dp)
            )

            // Record button
            RecordButton(
                isRecording = uiState.isRecording,
                captureMode = uiState.captureMode,
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

            // Switch camera button
            if (uiState.status == TrackingStatus.IDLE) {
                Button(
                    onClick = { viewModel.switchCamera() },
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(top = 64.dp, start = 16.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color.Black.copy(alpha = 0.5f))
                ) {
                    Text("\u21BB", color = Color.White, fontSize = 18.sp)
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
private fun TrackingOverlay(
    state: TrackingUiState,
    bracketOpacity: Float,
    lockScale: Float,
    reacquireColorBlend: Float,
    lostOpacity: Float
) {
    val isLocked = state.status == TrackingStatus.LOCKED
    val isLost = state.status == TrackingStatus.LOST
    val isIdle = state.status == TrackingStatus.IDLE
    val hasContour = state.lockedContour.size >= 3

    Canvas(modifier = Modifier.fillMaxSize()) {
        val transform = computeFillCenterTransform(
            size.width, size.height,
            state.sourceImageWidth, state.sourceImageHeight
        )

        if (isIdle) {
            state.detectedObjects.forEach { obj ->
                drawIdleBrackets(obj.boundingBox, transform, bracketOpacity)
            }
        } else if (isLocked && state.trackedObject != null) {
            val color = lerp(HapticGreen, HapticCyan, reacquireColorBlend)
                .copy(alpha = bracketOpacity)

            val box = state.trackedObject.boundingBox
            val (left, top, right, bottom) = mapBox(box, transform)
            val cx = (left + right) / 2f
            val cy = (top + bottom) / 2f

            scale(lockScale, pivot = Offset(cx, cy)) {
                drawRoundedGlow(left, top, right, bottom, color)
            }
        } else if (isLost && state.trackedObject != null && lostOpacity > 0.01f) {
            val color = HapticRed.copy(alpha = lostOpacity * 0.7f)
            val (ll, lt, lr, lb) = mapBox(state.trackedObject.boundingBox, transform)
            drawRoundedGlow(ll, lt, lr, lb, color)
        }
    }
}

// Cached Paint objects for glow rendering — avoids allocation on every frame.
// Color/alpha updated per-frame; BlurMaskFilter is reused.
private val outerGlowPaint = android.graphics.Paint().apply {
    style = android.graphics.Paint.Style.STROKE
    isAntiAlias = true
    strokeWidth = 30f
    maskFilter = android.graphics.BlurMaskFilter(40f, android.graphics.BlurMaskFilter.Blur.NORMAL)
}

private val innerGlowPaint = android.graphics.Paint().apply {
    style = android.graphics.Paint.Style.STROKE
    isAntiAlias = true
    strokeWidth = 15f
    maskFilter = android.graphics.BlurMaskFilter(20f, android.graphics.BlurMaskFilter.Blur.NORMAL)
}

/**
 * Draw a soft glowing rounded rectangle — backlight effect around the object.
 * Thick blurred strokes dissolve into diffused light. No fill = interior stays clear.
 */
private fun DrawScope.drawRoundedGlow(
    left: Float, top: Float, right: Float, bottom: Float,
    color: Color
) {
    val nativeCanvas = drawContext.canvas.nativeCanvas
    val w = right - left
    val h = bottom - top
    val cornerRadius = minOf(w, h) * 0.25f
    val rect = android.graphics.RectF(left, top, right, bottom)

    outerGlowPaint.color = color.copy(alpha = color.alpha * 0.20f).toArgb()
    nativeCanvas.drawRoundRect(rect, cornerRadius, cornerRadius, outerGlowPaint)

    innerGlowPaint.color = color.copy(alpha = color.alpha * 0.30f).toArgb()
    nativeCanvas.drawRoundRect(rect, cornerRadius, cornerRadius, innerGlowPaint)
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
 * Draw brackets from pre-computed screen coordinates with shadow.
 */
private fun DrawScope.drawBracketsRaw(
    left: Float, top: Float, right: Float, bottom: Float,
    color: Color
) {
    val w = right - left
    val h = bottom - top
    val cornerLen = minOf(w, h) * 0.2f
    val stroke = scaledStroke(w, h)

    // Shadow pass
    val shadow = Color.Black.copy(alpha = color.alpha * 0.4f)
    val so = 1.5f
    drawBracketLines(left + so, top + so, right + so, bottom + so, cornerLen, shadow, stroke)

    // Color pass
    drawBracketLines(left, top, right, bottom, cornerLen, color, stroke)
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
    drawBracketsRaw(left, top, right, bottom, color)
}

/**
 * Draw dashed corner brackets (for lost state — fading out).
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
    drawLine(color, Offset(left, top), Offset(left + cornerLen, top), strokeWidth, pathEffect = pathEffect)
    drawLine(color, Offset(left, top), Offset(left, top + cornerLen), strokeWidth, pathEffect = pathEffect)
    drawLine(color, Offset(right, top), Offset(right - cornerLen, top), strokeWidth, pathEffect = pathEffect)
    drawLine(color, Offset(right, top), Offset(right, top + cornerLen), strokeWidth, pathEffect = pathEffect)
    drawLine(color, Offset(left, bottom), Offset(left + cornerLen, bottom), strokeWidth, pathEffect = pathEffect)
    drawLine(color, Offset(left, bottom), Offset(left, bottom - cornerLen), strokeWidth, pathEffect = pathEffect)
    drawLine(color, Offset(right, bottom), Offset(right - cornerLen, bottom), strokeWidth, pathEffect = pathEffect)
    drawLine(color, Offset(right, bottom), Offset(right, bottom - cornerLen), strokeWidth, pathEffect = pathEffect)
}

/**
 * Draw thin corner brackets for idle detections — shows what's tappable.
 */
private fun DrawScope.drawIdleBrackets(box: RectF, transform: FillCenterTransform, opacity: Float) {
    val (left, top, right, bottom) = mapBox(box, transform)
    val w = right - left
    val h = bottom - top
    val cornerLen = minOf(w, h) * 0.15f
    val stroke = scaledStroke(w, h) * 0.6f
    val color = Color.White.copy(alpha = 0.5f * opacity)

    drawBracketLines(left, top, right, bottom, cornerLen, color, stroke)
}

// ---------------------------------------------------------------------------
// Label Overlay — locked object only (kept for testing, remove later)
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
        val boxHeightPx = bottom - transform.toScreenY(obj.boundingBox.top)

        val fontSize = (boxHeightPx * 0.06f).coerceIn(
            with(density) { 10.sp.toPx() },
            with(density) { 16.sp.toPx() }
        )
        val fontSizeSp = with(density) { fontSize.toSp() }

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
private fun RecordButton(
    isRecording: Boolean,
    captureMode: CaptureMode = CaptureMode.VIDEO,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Button(
        onClick = onClick,
        modifier = modifier.size(72.dp),
        shape = CircleShape,
        colors = ButtonDefaults.buttonColors(
            containerColor = when {
                isRecording -> HapticRed
                captureMode == CaptureMode.PHOTO -> Color.White
                else -> Color.White
            }
        )
    ) {
        if (isRecording) {
            Canvas(modifier = Modifier.size(24.dp)) {
                drawRect(color = Color.White)
            }
        } else if (captureMode == CaptureMode.PHOTO) {
            // Inner circle for photo mode (camera shutter style)
            Canvas(modifier = Modifier.size(28.dp)) {
                drawCircle(color = Color.White)
                drawCircle(color = Color.Black.copy(alpha = 0.15f), radius = size.minDimension / 2f * 0.85f)
            }
        }
    }
}

@Composable
private fun CaptureModePill(
    captureMode: CaptureMode,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier
) {
    var totalDrag by remember { mutableStateOf(0f) }

    Row(
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.4f), RoundedCornerShape(16.dp))
            .padding(horizontal = 4.dp, vertical = 2.dp)
            .pointerInput(Unit) {
                detectHorizontalDragGestures(
                    onDragEnd = {
                        if (abs(totalDrag) > 50f) {
                            onToggle()
                        }
                        totalDrag = 0f
                    },
                    onHorizontalDrag = { _, dragAmount ->
                        totalDrag += dragAmount
                    }
                )
            },
        horizontalArrangement = Arrangement.spacedBy(2.dp)
    ) {
        ModeLabel("VIDEO", selected = captureMode == CaptureMode.VIDEO)
        ModeLabel("PHOTO", selected = captureMode == CaptureMode.PHOTO)
    }
}

@Composable
private fun ModeLabel(text: String, selected: Boolean) {
    Text(
        text = text,
        color = if (selected) Color.White else Color.White.copy(alpha = 0.5f),
        fontSize = 12.sp,
        fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
        modifier = Modifier
            .background(
                if (selected) Color.White.copy(alpha = 0.15f) else Color.Transparent,
                RoundedCornerShape(12.dp)
            )
            .padding(horizontal = 12.dp, vertical = 4.dp)
    )
}
