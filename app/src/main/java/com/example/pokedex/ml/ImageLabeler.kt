package com.example.pokedex.ml

import android.content.Context
import android.net.Uri
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import kotlinx.coroutines.tasks.await

data class LabelResult(val text: String, val confidence: Float)

class ImageLabeler {
    private val client = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS)

    suspend fun labelFromUri(context: Context, uri: Uri): List<LabelResult> {
        val image = InputImage.fromFilePath(context, uri)
        val labels = client.process(image).await()
        return labels
            .map { LabelResult(it.text, it.confidence) }
            .sortedByDescending { it.confidence }
    }
}
