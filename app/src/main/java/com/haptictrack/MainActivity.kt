package com.haptictrack

import android.os.Bundle
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import com.haptictrack.ui.CameraScreen
import com.haptictrack.ui.CameraViewModel
import com.haptictrack.ui.theme.HapticTrackTheme

class MainActivity : ComponentActivity() {

    private val viewModel: CameraViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            HapticTrackTheme {
                CameraScreen(viewModel)
            }
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN) {
            viewModel.onVolumeDown()
            return true
        }
        if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            viewModel.onVolumeUp()
            return true
        }
        return super.onKeyDown(keyCode, event)
    }
}
