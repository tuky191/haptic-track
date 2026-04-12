package com.haptictrack.ui

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
                            val normalizedX = event.x / previewView.width
                            val normalizedY = event.y / previewView.height
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
        // Draw all detected objects
        state.detectedObjects.forEach { obj ->
            val isLocked = obj.id == lockedId && state.status == TrackingStatus.LOCKED
            val rect = obj.boundingBox
            val color = if (isLocked) HapticGreen else Color.White.copy(alpha = 0.5f)
            val strokeWidth = if (isLocked) 4f else 2f

            drawRect(
                color = color,
                topLeft = Offset(rect.left * size.width, rect.top * size.height),
                size = Size(rect.width() * size.width, rect.height() * size.height),
                style = Stroke(width = strokeWidth)
            )

            // Corner accents for locked object
            if (isLocked) {
                val cornerLen = minOf(rect.width() * size.width, rect.height() * size.height) * 0.2f
                val left = rect.left * size.width
                val top = rect.top * size.height
                val right = rect.right * size.width
                val bottom = rect.bottom * size.height

                drawLine(HapticGreen, Offset(left, top), Offset(left + cornerLen, top), 6f)
                drawLine(HapticGreen, Offset(left, top), Offset(left, top + cornerLen), 6f)
                drawLine(HapticGreen, Offset(right, top), Offset(right - cornerLen, top), 6f)
                drawLine(HapticGreen, Offset(right, top), Offset(right, top + cornerLen), 6f)
                drawLine(HapticGreen, Offset(left, bottom), Offset(left + cornerLen, bottom), 6f)
                drawLine(HapticGreen, Offset(left, bottom), Offset(left, bottom - cornerLen), 6f)
                drawLine(HapticGreen, Offset(right, bottom), Offset(right - cornerLen, bottom), 6f)
                drawLine(HapticGreen, Offset(right, bottom), Offset(right, bottom - cornerLen), 6f)
            }
        }

        // Draw last-known box for lost object
        if (state.status == TrackingStatus.LOST && state.trackedObject != null) {
            val rect = state.trackedObject.boundingBox
            drawRect(
                color = HapticRed.copy(alpha = 0.5f),
                topLeft = Offset(rect.left * size.width, rect.top * size.height),
                size = Size(rect.width() * size.width, rect.height() * size.height),
                style = Stroke(width = 3f)
            )
        }
    }
}

@Composable
private fun ObjectLabelOverlay(state: TrackingUiState) {
    val lockedId = state.trackedObject?.id

    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val parentWidth = constraints.maxWidth.toFloat()
        val parentHeight = constraints.maxHeight.toFloat()
        val density = LocalDensity.current

        state.detectedObjects.forEach { obj ->
            val labelText = formatObjectLabel(obj)
            val isLocked = obj.id == lockedId && state.status == TrackingStatus.LOCKED
            val bgColor = if (isLocked) HapticGreen else Color.Black.copy(alpha = 0.6f)
            val textColor = Color.White

            val xPx = obj.boundingBox.left * parentWidth
            val yPx = (obj.boundingBox.top * parentHeight) - with(density) { 20.dp.toPx() }

            Text(
                text = labelText,
                color = textColor,
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
            val xPx = obj.boundingBox.left * parentWidth
            val yPx = (obj.boundingBox.top * parentHeight) - with(density) { 20.dp.toPx() }

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
    val label = obj.label
    if (label == null) return "Object #${obj.id}"
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