package com.haptictrack.ui

import android.graphics.RectF
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import androidx.camera.view.PreviewView
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.pointerInteropFilter
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.rememberMultiplePermissionsState
import com.haptictrack.tracking.CaptureMode
import com.haptictrack.tracking.TrackingFilter
import com.haptictrack.tracking.TrackingStatus
import com.haptictrack.tracking.labelMatchesFilter
import com.haptictrack.tracking.TrackingUiState
import com.haptictrack.ui.theme.HapticAmber
import com.haptictrack.ui.theme.HapticCyan
import com.haptictrack.ui.theme.HapticGreen
import com.haptictrack.ui.theme.HapticRed
import kotlin.math.abs
import kotlinx.coroutines.delay

// ---------------------------------------------------------------------------
// Coordinate Transform (FILL_CENTER)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Main Screen
// ---------------------------------------------------------------------------

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

        val lockPulse = remember { Animatable(1f) }
        var previousStatus by remember { mutableStateOf(TrackingStatus.IDLE) }
        var previousLockedId by remember { mutableStateOf<Int?>(null) }

        var reacquireBrightness by remember { mutableStateOf(0f) }
        val reacquireColorBlend by animateFloatAsState(
            targetValue = reacquireBrightness,
            animationSpec = tween(300, easing = FastOutSlowInEasing),
            label = "reacquireColor"
        )

        val lostOpacity by animateFloatAsState(
            targetValue = if (uiState.status == TrackingStatus.LOST) 0f else 1f,
            animationSpec = tween(
                durationMillis = if (uiState.status == TrackingStatus.LOST) 2000 else 0,
                easing = LinearEasing
            ),
            label = "lostFade"
        )

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
                lockPulse.snapTo(0.92f)
                lockPulse.animateTo(1f, tween(200, easing = FastOutSlowInEasing))
            }

            if (justReacquired) {
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
                zoomIndicatorOpacity.animateTo(0f, tween(500, easing = FastOutSlowInEasing))
            }
        }

        // --- Recording timer ---
        var elapsedSeconds by remember { mutableIntStateOf(0) }
        LaunchedEffect(uiState.isRecording) {
            if (uiState.isRecording) {
                elapsedSeconds = 0
                while (true) {
                    delay(1000L)
                    elapsedSeconds++
                }
            } else {
                elapsedSeconds = 0
            }
        }

        // --- Pinch-to-zoom ---
        val scaleDetector = remember {
            ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    viewModel.onPinchZoom(detector.scaleFactor)
                    return true
                }
                override fun onScaleEnd(detector: ScaleGestureDetector) {
                    viewModel.hideZoomIndicator()
                }
            })
        }
        var isScaling by remember { mutableStateOf(false) }

        // --- Debug sheet ---
        var showDebugSheet by remember { mutableStateOf(false) }

        Box(modifier = Modifier.fillMaxSize()) {
            // Camera preview (hidden — surface routed to SurfaceTextureFrameReader)
            AndroidView(
                factory = { previewView },
                modifier = Modifier
                    .fillMaxSize()
                    .pointerInteropFilter { event ->
                        if (uiState.stealthMode) return@pointerInteropFilter true

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

            // Viewfinder from SurfaceTexture GL thread
            if (!uiState.stealthMode) {
                val viewfinderBmp by viewModel.viewfinderBitmap.collectAsStateWithLifecycle()
                viewfinderBmp?.let { bmp ->
                    Image(
                        bitmap = bmp.asImageBitmap(),
                        contentDescription = null,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop
                    )
                }
            }

            // Stealth overlay
            if (uiState.stealthMode) {
                Box(modifier = Modifier.fillMaxSize().background(Color.Black))
            }

            // Loading overlay
            if (!uiState.isReady) {
                Box(
                    modifier = Modifier.fillMaxSize().background(Color.Black),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(color = Color.White, strokeWidth = 2.dp)
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(uiState.loadingStatus, color = Color.White.copy(alpha = 0.7f), fontSize = 13.sp)
                    }
                }
            }

            if (!uiState.stealthMode) {
                // Tracking overlay (bounding boxes)
                TrackingOverlay(
                    state = uiState,
                    bracketOpacity = bracketOpacity,
                    lockScale = lockPulse.value,
                    reacquireColorBlend = reacquireColorBlend,
                    lostOpacity = lostOpacity,
                    trackingFilter = uiState.trackingFilter
                )

                // Top HUD
                val showZoomAlways = uiState.status == TrackingStatus.LOCKED || uiState.status == TrackingStatus.LOST
                val zoomAlpha = if (showZoomAlways) 1f else zoomIndicatorOpacity.value

                Column(
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .fillMaxWidth()
                        .statusBarsPadding(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    TopBar(
                        isRecording = uiState.isRecording,
                        elapsedSeconds = elapsedSeconds,
                        zoomRatio = uiState.currentZoomRatio,
                        zoomAlpha = zoomAlpha,
                        onSettingsClick = { showDebugSheet = true }
                    )
                    StatusIndicator(
                        status = uiState.status,
                        label = uiState.trackedObject?.label
                    )
                }

                // Bottom controls
                BottomControls(
                    isRecording = uiState.isRecording,
                    captureMode = uiState.captureMode,
                    trackingFilter = uiState.trackingFilter,
                    showFlip = uiState.status == TrackingStatus.IDLE,
                    showClear = uiState.status != TrackingStatus.IDLE,
                    isIdle = uiState.status == TrackingStatus.IDLE,
                    onShutterClick = { viewModel.toggleRecording() },
                    onFlipClick = { viewModel.switchCamera() },
                    onClearClick = { viewModel.clearTracking() },
                    onModeToggle = { viewModel.toggleCaptureMode() },
                    onFilterCycle = { viewModel.cycleTrackingFilter() },
                    modifier = Modifier.align(Alignment.BottomCenter)
                )

                // Debug bottom sheet
                if (showDebugSheet) {
                    DebugBottomSheet(
                        uiState = uiState,
                        onToggleIsp = { viewModel.toggleIspStabilization() },
                        onToggleGyro = { viewModel.toggleGyroEis() },
                        onGyroStrengthChange = { viewModel.setGyroStrength(it) },
                        onToggleAdaptive = { viewModel.toggleAdaptiveEis() },
                        onToggleTranslation = { viewModel.toggleTranslationEis() },
                        onToggleLeash = { viewModel.toggleLeash() },
                        onToggleOis = { viewModel.toggleOisCompensation() },
                        onHapticStrengthChange = { viewModel.setHapticStrength(it) },
                        onDismiss = { showDebugSheet = false }
                    )
                }
            }
        }
    } else {
        LaunchedEffect(Unit) {
            permissions.launchMultiplePermissionRequest()
        }
        Box(modifier = Modifier.fillMaxSize().background(Color.Black), contentAlignment = Alignment.Center) {
            Text("Camera and microphone permissions required", color = Color.White.copy(alpha = 0.7f), fontSize = 14.sp)
        }
    }
}

// ---------------------------------------------------------------------------
// Top Bar
// ---------------------------------------------------------------------------

@Composable
private fun TopBar(
    isRecording: Boolean,
    elapsedSeconds: Int,
    zoomRatio: Float,
    zoomAlpha: Float,
    onSettingsClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Left: recording indicator or empty space
        Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.CenterStart) {
            if (isRecording) {
                RecordingIndicator(elapsedSeconds)
            }
        }

        // Center: zoom pill
        ZoomPill(zoomRatio, zoomAlpha)

        // Right: settings gear
        Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.CenterEnd) {
            CircleIconButton(
                icon = Icons.Filled.Settings,
                contentDescription = "Settings",
                onClick = onSettingsClick,
                size = 40.dp
            )
        }
    }
}

@Composable
private fun RecordingIndicator(elapsedSeconds: Int) {
    val dotAlpha = remember { Animatable(1f) }
    LaunchedEffect(Unit) {
        while (true) {
            dotAlpha.animateTo(0.3f, tween(500))
            dotAlpha.animateTo(1f, tween(500))
        }
    }

    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .background(Color.Black.copy(alpha = 0.4f), RoundedCornerShape(16.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp)
    ) {
        Canvas(Modifier.size(8.dp)) {
            drawCircle(HapticRed.copy(alpha = dotAlpha.value))
        }
        Spacer(Modifier.width(6.dp))
        Text(
            text = "%02d:%02d".format(elapsedSeconds / 60, elapsedSeconds % 60),
            color = Color.White,
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium,
            fontFamily = FontFamily.Monospace
        )
    }
}

@Composable
private fun ZoomPill(zoomRatio: Float, alpha: Float) {
    Text(
        text = "%.1f×".format(zoomRatio),
        color = Color.White.copy(alpha = alpha),
        fontSize = 14.sp,
        fontWeight = FontWeight.SemiBold,
        modifier = Modifier
            .background(
                Color.Black.copy(alpha = 0.4f * alpha),
                RoundedCornerShape(16.dp)
            )
            .padding(horizontal = 14.dp, vertical = 6.dp)
    )
}

// ---------------------------------------------------------------------------
// Status Indicator
// ---------------------------------------------------------------------------

@Composable
private fun StatusIndicator(
    status: TrackingStatus,
    label: String?,
    modifier: Modifier = Modifier
) {
    val (text, color) = when (status) {
        TrackingStatus.IDLE -> "Tap to track" to Color.White.copy(alpha = 0.5f)
        TrackingStatus.SEARCHING -> "Searching" to HapticAmber
        TrackingStatus.LOCKED -> (if (label != null) "Tracking · $label" else "Tracking") to HapticGreen
        TrackingStatus.LOST -> "Lost" to HapticRed
    }

    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = modifier
            .padding(top = 2.dp)
            .background(Color.Black.copy(alpha = 0.25f), RoundedCornerShape(12.dp))
            .padding(horizontal = 10.dp, vertical = 4.dp)
    ) {
        if (status != TrackingStatus.IDLE) {
            Canvas(Modifier.size(6.dp)) {
                drawCircle(color)
            }
            Spacer(Modifier.width(6.dp))
        }
        Text(
            text = text,
            color = color,
            fontSize = 12.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

// ---------------------------------------------------------------------------
// Bottom Controls
// ---------------------------------------------------------------------------

@Composable
private fun BottomControls(
    isRecording: Boolean,
    captureMode: CaptureMode,
    trackingFilter: TrackingFilter,
    showFlip: Boolean,
    showClear: Boolean,
    isIdle: Boolean,
    onShutterClick: () -> Unit,
    onFlipClick: () -> Unit,
    onClearClick: () -> Unit,
    onModeToggle: () -> Unit,
    onFilterCycle: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .padding(bottom = 24.dp)
    ) {
        // Tracking filter (IDLE only)
        if (isIdle && !isRecording) {
            TrackingFilterPill(trackingFilter, onFilterCycle)
            Spacer(Modifier.height(8.dp))
        }

        // Mode selector
        if (!isRecording) {
            CaptureModePill(captureMode, onModeToggle)
            Spacer(Modifier.height(20.dp))
        } else {
            Spacer(Modifier.height(52.dp))
        }

        // Controls row: [flip] [shutter] [clear]
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 48.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (showFlip) {
                CircleIconButton(
                    icon = Icons.Filled.Refresh,
                    contentDescription = "Switch camera",
                    onClick = onFlipClick
                )
            } else {
                Spacer(Modifier.size(48.dp))
            }

            ShutterButton(
                isRecording = isRecording,
                onClick = onShutterClick
            )

            if (showClear) {
                CircleIconButton(
                    icon = Icons.Filled.Close,
                    contentDescription = "Clear tracking",
                    onClick = onClearClick
                )
            } else {
                Spacer(Modifier.size(48.dp))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shutter Button
// ---------------------------------------------------------------------------

@Composable
private fun ShutterButton(
    isRecording: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val innerSize by animateDpAsState(
        targetValue = if (isRecording) 24.dp else 56.dp,
        animationSpec = tween(200, easing = FastOutSlowInEasing),
        label = "shutterInner"
    )
    val innerCorner by animateDpAsState(
        targetValue = if (isRecording) 6.dp else 28.dp,
        animationSpec = tween(200, easing = FastOutSlowInEasing),
        label = "shutterCorner"
    )
    val innerColor by animateColorAsState(
        targetValue = if (isRecording) HapticRed else Color.White,
        animationSpec = tween(200),
        label = "shutterColor"
    )

    Box(
        modifier = modifier
            .size(72.dp)
            .clip(CircleShape)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        // Outer ring
        Box(
            Modifier
                .fillMaxSize()
                .border(4.dp, Color.White.copy(alpha = 0.9f), CircleShape)
        )
        // Inner shape (circle when idle, rounded square when recording)
        Box(
            Modifier
                .size(innerSize)
                .background(innerColor, RoundedCornerShape(innerCorner))
        )
    }
}

// ---------------------------------------------------------------------------
// Capture Mode Pill
// ---------------------------------------------------------------------------

@Composable
private fun CaptureModePill(
    captureMode: CaptureMode,
    onToggle: () -> Unit,
    modifier: Modifier = Modifier
) {
    var totalDrag by remember { mutableStateOf(0f) }

    Row(
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.3f), RoundedCornerShape(16.dp))
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
        color = if (selected) Color.White else Color.White.copy(alpha = 0.4f),
        fontSize = 12.sp,
        fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
        modifier = Modifier
            .background(
                if (selected) Color.White.copy(alpha = 0.12f) else Color.Transparent,
                RoundedCornerShape(12.dp)
            )
            .padding(horizontal = 14.dp, vertical = 4.dp)
    )
}

// ---------------------------------------------------------------------------
// Tracking Filter Pill
// ---------------------------------------------------------------------------

@Composable
private fun TrackingFilterPill(
    filter: TrackingFilter,
    onCycle: () -> Unit,
    modifier: Modifier = Modifier
) {
    val label = when (filter) {
        TrackingFilter.ALL -> "All"
        TrackingFilter.PERSON_ONLY -> "People"
        TrackingFilter.PETS -> "Pets"
        TrackingFilter.NON_PERSON_ONLY -> "Things"
    }

    Text(
        text = label,
        color = when (filter) {
            TrackingFilter.ALL -> Color.White.copy(alpha = 0.6f)
            TrackingFilter.PERSON_ONLY -> HapticGreen
            TrackingFilter.PETS -> HapticCyan
            TrackingFilter.NON_PERSON_ONLY -> HapticAmber
        },
        fontSize = 12.sp,
        fontWeight = FontWeight.Medium,
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .clickable(onClick = onCycle)
            .background(Color.Black.copy(alpha = 0.3f), RoundedCornerShape(12.dp))
            .padding(horizontal = 14.dp, vertical = 4.dp)
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

@Composable
private fun CircleIconButton(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    size: Dp = 48.dp,
    tint: Color = Color.White
) {
    IconButton(
        onClick = onClick,
        modifier = modifier
            .size(size)
            .background(Color.Black.copy(alpha = 0.4f), CircleShape)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = contentDescription,
            tint = tint,
            modifier = Modifier.size(size * 0.5f)
        )
    }
}

// ---------------------------------------------------------------------------
// Tracking Overlay
// ---------------------------------------------------------------------------

@Composable
private fun TrackingOverlay(
    state: TrackingUiState,
    bracketOpacity: Float,
    lockScale: Float,
    reacquireColorBlend: Float,
    lostOpacity: Float,
    trackingFilter: TrackingFilter = TrackingFilter.ALL
) {
    val isLocked = state.status == TrackingStatus.LOCKED
    val isLost = state.status == TrackingStatus.LOST
    val isIdle = state.status == TrackingStatus.IDLE

    Canvas(modifier = Modifier.fillMaxSize()) {
        val transform = computeFillCenterTransform(
            size.width, size.height,
            state.sourceImageWidth, state.sourceImageHeight
        )

        if (isIdle) {
            state.detectedObjects
                .filter { labelMatchesFilter(it.label, trackingFilter) }
                .forEach { obj ->
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

private fun scaledStroke(boxWidthPx: Float, boxHeightPx: Float): Float {
    return (minOf(boxWidthPx, boxHeightPx) * 0.008f).coerceIn(2f, 6f)
}

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

private fun DrawScope.drawBracketLines(
    left: Float, top: Float, right: Float, bottom: Float,
    cornerLen: Float, color: Color, strokeWidth: Float
) {
    drawLine(color, Offset(left, top), Offset(left + cornerLen, top), strokeWidth)
    drawLine(color, Offset(left, top), Offset(left, top + cornerLen), strokeWidth)
    drawLine(color, Offset(right, top), Offset(right - cornerLen, top), strokeWidth)
    drawLine(color, Offset(right, top), Offset(right, top + cornerLen), strokeWidth)
    drawLine(color, Offset(left, bottom), Offset(left + cornerLen, bottom), strokeWidth)
    drawLine(color, Offset(left, bottom), Offset(left, bottom - cornerLen), strokeWidth)
    drawLine(color, Offset(right, bottom), Offset(right - cornerLen, bottom), strokeWidth)
    drawLine(color, Offset(right, bottom), Offset(right, bottom - cornerLen), strokeWidth)
}

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
// Debug Bottom Sheet
// ---------------------------------------------------------------------------

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DebugBottomSheet(
    uiState: TrackingUiState,
    onToggleIsp: () -> Unit,
    onToggleGyro: () -> Unit,
    onGyroStrengthChange: (Float) -> Unit,
    onToggleAdaptive: () -> Unit,
    onToggleTranslation: () -> Unit,
    onToggleLeash: () -> Unit,
    onToggleOis: () -> Unit,
    onHapticStrengthChange: (Float) -> Unit,
    onDismiss: () -> Unit
) {
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = Color(0xFF1A1A1A),
        contentColor = Color.White
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 32.dp)
        ) {
            Text(
                text = "STABILIZATION",
                color = Color.White.copy(alpha = 0.4f),
                fontSize = 11.sp,
                fontWeight = FontWeight.SemiBold,
                letterSpacing = 1.5.sp,
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 8.dp)
            )

            SettingRow("ISP Stabilization", uiState.ispStabilization, onToggleIsp)
            SettingRow("Gyro EIS", uiState.gyroEis, onToggleGyro)

            if (uiState.gyroEis) {
                // Strength slider
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 24.dp, vertical = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "Strength",
                        color = Color.White,
                        fontSize = 14.sp,
                        modifier = Modifier.weight(0.35f)
                    )
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.weight(0.65f)
                    ) {
                        Text("Lo", color = Color.White.copy(alpha = 0.35f), fontSize = 11.sp)
                        Slider(
                            value = uiState.gyroStrength,
                            onValueChange = onGyroStrengthChange,
                            modifier = Modifier.weight(1f).height(32.dp),
                            colors = SliderDefaults.colors(
                                thumbColor = Color.White,
                                activeTrackColor = Color.White.copy(alpha = 0.5f),
                                inactiveTrackColor = Color.White.copy(alpha = 0.12f)
                            )
                        )
                        Text("Hi", color = Color.White.copy(alpha = 0.35f), fontSize = 11.sp)
                    }
                }

                SettingRow("Adaptive Smoothing", uiState.adaptiveEis, onToggleAdaptive)
                SettingRow("Translation Correction", uiState.translationEis, onToggleTranslation)
                SettingRow("Leash", uiState.leashEnabled, onToggleLeash)
                SettingRow("OIS Compensation", uiState.oisCompensation, onToggleOis)
            }

            Spacer(Modifier.height(8.dp))

            Text(
                text = "HAPTICS",
                color = Color.White.copy(alpha = 0.4f),
                fontSize = 11.sp,
                fontWeight = FontWeight.SemiBold,
                letterSpacing = 1.5.sp,
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 8.dp)
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "Strength",
                    color = Color.White,
                    fontSize = 14.sp,
                    modifier = Modifier.weight(0.35f)
                )
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.weight(0.65f)
                ) {
                    Text("Off", color = Color.White.copy(alpha = 0.35f), fontSize = 11.sp)
                    Slider(
                        value = uiState.hapticStrength,
                        onValueChange = onHapticStrengthChange,
                        modifier = Modifier.weight(1f).height(32.dp),
                        colors = SliderDefaults.colors(
                            thumbColor = Color.White,
                            activeTrackColor = Color.White.copy(alpha = 0.5f),
                            inactiveTrackColor = Color.White.copy(alpha = 0.12f)
                        )
                    )
                    Text("Max", color = Color.White.copy(alpha = 0.35f), fontSize = 11.sp)
                }
            }
        }
    }
}

@Composable
private fun SettingRow(label: String, enabled: Boolean, onToggle: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onToggle)
            .padding(horizontal = 24.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, color = Color.White, fontSize = 14.sp)
        Switch(
            checked = enabled,
            onCheckedChange = { onToggle() },
            colors = SwitchDefaults.colors(
                checkedThumbColor = HapticGreen,
                checkedTrackColor = HapticGreen.copy(alpha = 0.3f),
                uncheckedThumbColor = Color.White.copy(alpha = 0.5f),
                uncheckedTrackColor = Color.White.copy(alpha = 0.12f),
                uncheckedBorderColor = Color.Transparent
            )
        )
    }
}
