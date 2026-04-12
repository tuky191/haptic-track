package com.haptictrack.ui

import android.graphics.RectF
import android.view.MotionEvent
import androidx.camera.view.PreviewView
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
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
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
import com.haptictrack.ui.theme.HapticGreen
import com.haptictrack.ui.theme.HapticRed

/**
 * Computes the FILL_CENTER transform: scale + offset to map normalized image
 * coordinates (0..1) to screen pixel coordinates.
 *
 * FILL_CENTER scales the image so the shorter dimension fills the view,
 * then crops the longer dimension equally on both sides.
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

    // FILL_CENTER: scale so the image fully covers the view, then crop
    val scale = if (imageAspect > viewAspect) {
        // Image is wider than view → match heights, crop width
        viewHeight / imageHeight
    } else {
        // Image is taller than view → match widths, crop height
        viewWidth / imageWidth
    }

    val mappedWidth = imageWidth * scale
    val mappedHeight = imageHeight * scale
    val offsetX = (viewWidth - mappedWidth) / 2f
    val offsetY = (viewHeight - mappedHeight) / 2f

    return FillCenterTransform(scale, offsetX, offsetY, mappedWidth, mappedHeight)
}

/** Map a normalized image coordinate to screen pixel coordinate. */
private fun FillCenterTransform.toScreenX(normalizedX: Float): Float =
    offsetX + normalizedX * mappedWidth

private fun FillCenterTransform.toScreenY(normalizedY: Float): Float =
    offsetY + normalizedY * mappedHeight

/** Map a screen pixel coordinate back to normalized image coordinate. */
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

            // Bounding box overlay (Canvas)
            BoundingBoxOverlay(uiState)

            // Labels overlay (Compose Text elements positioned over boxes)
            ObjectLabelOverlay(uiState)

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

@Composable
private fun BoundingBoxOverlay(state: TrackingUiState) {
    val lockedId = state.trackedObject?.id
    Canvas(modifier = Modifier.fillMaxSize()) {
        val transform = computeFillCenterTransform(
            size.width, size.height,
            state.sourceImageWidth, state.sourceImageHeight
        )

        state.detectedObjects.forEach { obj ->
            val isLocked = obj.id == lockedId && state.status == TrackingStatus.LOCKED
            val color = if (isLocked) HapticGreen else Color.White.copy(alpha = 0.5f)
            val strokeWidth = if (isLocked) 4f else 2f

            drawMappedRect(obj.boundingBox, transform, color, strokeWidth)

            if (isLocked) {
                drawCornerAccents(obj.boundingBox, transform)
            }
        }

        // Draw last-known box for lost object
        if (state.status == TrackingStatus.LOST && state.trackedObject != null) {
            drawMappedRect(state.trackedObject.boundingBox, transform, HapticRed.copy(alpha = 0.5f), 3f)
        }
    }
}

private fun DrawScope.drawMappedRect(
    box: RectF,
    transform: FillCenterTransform,
    color: Color,
    strokeWidth: Float
) {
    val left = transform.toScreenX(box.left)
    val top = transform.toScreenY(box.top)
    val right = transform.toScreenX(box.right)
    val bottom = transform.toScreenY(box.bottom)

    drawRect(
        color = color,
        topLeft = Offset(left, top),
        size = Size(right - left, bottom - top),
        style = Stroke(width = strokeWidth)
    )
}

private fun DrawScope.drawCornerAccents(box: RectF, transform: FillCenterTransform) {
    val left = transform.toScreenX(box.left)
    val top = transform.toScreenY(box.top)
    val right = transform.toScreenX(box.right)
    val bottom = transform.toScreenY(box.bottom)
    val cornerLen = minOf(right - left, bottom - top) * 0.2f

    drawLine(HapticGreen, Offset(left, top), Offset(left + cornerLen, top), 6f)
    drawLine(HapticGreen, Offset(left, top), Offset(left, top + cornerLen), 6f)
    drawLine(HapticGreen, Offset(right, top), Offset(right - cornerLen, top), 6f)
    drawLine(HapticGreen, Offset(right, top), Offset(right, top + cornerLen), 6f)
    drawLine(HapticGreen, Offset(left, bottom), Offset(left + cornerLen, bottom), 6f)
    drawLine(HapticGreen, Offset(left, bottom), Offset(left, bottom - cornerLen), 6f)
    drawLine(HapticGreen, Offset(right, bottom), Offset(right - cornerLen, bottom), 6f)
    drawLine(HapticGreen, Offset(right, bottom), Offset(right, bottom - cornerLen), 6f)
}

@Composable
private fun ObjectLabelOverlay(state: TrackingUiState) {
    val lockedId = state.trackedObject?.id

    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val parentWidth = constraints.maxWidth.toFloat()
        val parentHeight = constraints.maxHeight.toFloat()
        val density = LocalDensity.current
        val transform = computeFillCenterTransform(
            parentWidth, parentHeight,
            state.sourceImageWidth, state.sourceImageHeight
        )

        state.detectedObjects.forEach { obj ->
            val labelText = formatObjectLabel(obj)
            val isLocked = obj.id == lockedId && state.status == TrackingStatus.LOCKED
            val bgColor = if (isLocked) HapticGreen else Color.Black.copy(alpha = 0.6f)

            val xPx = transform.toScreenX(obj.boundingBox.left)
            val yPx = transform.toScreenY(obj.boundingBox.top) - with(density) { 20.dp.toPx() }

            Text(
                text = labelText,
                color = Color.White,
                fontSize = 11.sp,
                modifier = Modifier
                    .offset(
                        x = with(density) { xPx.toDp() },
                        y = with(density) { yPx.coerceAtLeast(0f).toDp() }
                    )
                    .background(bgColor, RoundedCornerShape(4.dp))
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            )
        }

        // Label for lost object
        if (state.status == TrackingStatus.LOST && state.trackedObject != null) {
            val obj = state.trackedObject
            val xPx = transform.toScreenX(obj.boundingBox.left)
            val yPx = transform.toScreenY(obj.boundingBox.top) - with(density) { 20.dp.toPx() }

            Text(
                text = "Lost: ${obj.label ?: "Object"}",
                color = Color.White,
                fontSize = 11.sp,
                modifier = Modifier
                    .offset(
                        x = with(density) { xPx.toDp() },
                        y = with(density) { yPx.coerceAtLeast(0f).toDp() }
                    )
                    .background(HapticRed.copy(alpha = 0.7f), RoundedCornerShape(4.dp))
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            )
        }
    }
}

private fun formatObjectLabel(obj: TrackedObject): String {
    val label = obj.label ?: return "Object #${obj.id}"
    val pct = (obj.confidence * 100).toInt()
    return if (pct > 0) "$label $pct%" else label
}

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
