package com.haptictrack

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.haptictrack.ui.CameraScreen
import com.haptictrack.ui.theme.HapticTrackTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            HapticTrackTheme {
                CameraScreen()
            }
        }
    }
}
