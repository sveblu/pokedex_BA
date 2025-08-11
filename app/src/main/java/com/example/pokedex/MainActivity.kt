package com.example.pokedex

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.example.pokedex.app.ImageLabelScreen
import com.example.pokedex.app.ObjectDetectionScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ImageLabelScreen()
            // ObjectDetectionScreen()
        }
    }
}
