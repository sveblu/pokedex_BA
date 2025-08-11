package com.example.pokedex.ml

import android.graphics.Rect

data class DetectionResult(
    val boundingBox: Rect,
    val category: String?,
    val confidence: Float?
)
