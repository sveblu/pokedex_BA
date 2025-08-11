package com.example.pokedex.ml

import android.content.Context
import android.net.Uri
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.DetectedObject
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import kotlinx.coroutines.tasks.await

class ObjectDetector(private val context: Context) {

    private val detector by lazy {
        val options = ObjectDetectorOptions.Builder()
            .setDetectorMode(ObjectDetectorOptions.SINGLE_IMAGE_MODE)
            .enableMultipleObjects()
            .enableClassification() // coarse categories like "fashion", "food"
            .build()
        ObjectDetection.getClient(options)
    }

    suspend fun detectFromUri(uri: Uri): List<DetectionResult> {
        val image = InputImage.fromFilePath(context, uri)
        val objects: List<DetectedObject> = detector.process(image).await()

        return objects.map { obj ->
            val category = obj.labels.firstOrNull()?.text
            val confidence = obj.labels.firstOrNull()?.confidence
            DetectionResult(
                boundingBox = obj.boundingBox,
                category = category,
                confidence = confidence
            )
        }
    }
}
