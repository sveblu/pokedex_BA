//package com.example.pokedex
//
//import android.util.Log
//import android.view.ViewGroup
//import androidx.annotation.OptIn
//import androidx.camera.core.*
//import androidx.camera.lifecycle.ProcessCameraProvider
//import androidx.camera.view.PreviewView
//import androidx.compose.foundation.background
//import androidx.compose.foundation.layout.Box
//import androidx.compose.foundation.layout.fillMaxSize
//import androidx.compose.foundation.layout.padding
//import androidx.compose.material3.Text
//import androidx.compose.runtime.*
//import androidx.compose.ui.Alignment
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.graphics.Color
//import androidx.compose.ui.platform.LocalContext
//import androidx.compose.ui.unit.dp
//import androidx.compose.ui.unit.sp
//import androidx.compose.ui.viewinterop.AndroidView
//import androidx.core.content.ContextCompat
//import com.google.mlkit.vision.common.InputImage
//import com.google.mlkit.vision.label.ImageLabeling
//import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
//import com.google.mlkit.vision.objects.DetectedObject
//import com.google.mlkit.vision.objects.ObjectDetection
//import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
//import java.util.concurrent.Executors
//
//@OptIn(ExperimentalGetImage::class)
//@Composable
//fun CameraScreen() {
//    val context = LocalContext.current
//    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
//    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
//
//    var topLabel by remember { mutableStateOf<String?>(null) }
//    var objectsCount by remember { mutableIntStateOf(0) }
//
//    Box(Modifier.fillMaxSize()) {
//        AndroidView(
//            modifier = Modifier.fillMaxSize(),
//            factory = { ctx ->
//                val previewView = PreviewView(ctx).apply {
//                    layoutParams = ViewGroup.LayoutParams(
//                        ViewGroup.LayoutParams.MATCH_PARENT,
//                        ViewGroup.LayoutParams.MATCH_PARENT
//                    )
//                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
//                }
//
//                cameraProviderFuture.addListener({
//                    val cameraProvider = cameraProviderFuture.get()
//                    val preview = Preview.Builder().build().also {
//                        it.surfaceProvider = previewView.surfaceProvider
//                    }
//
//                    val analysis = ImageAnalysis.Builder()
//                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//                        .build()
//
//                    val labeler = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS)
//
//                    val odtOptions = ObjectDetectorOptions.Builder()
//                        .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
//                        .enableMultipleObjects()
//                        .enableClassification()
//                        .build()
//                    val detector = ObjectDetection.getClient(odtOptions)
//
//                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
//                        val media = imageProxy.image
//                        if (media == null) {
//                            imageProxy.close(); return@setAnalyzer
//                        }
//                        val img = InputImage.fromMediaImage(
//                            media,
//                            imageProxy.imageInfo.rotationDegrees
//                        )
//
//                        // Image labeling
//                        labeler.process(img)
//                            .addOnSuccessListener { labels ->
//                                topLabel = labels.maxByOrNull { it.confidence }?.text
//                            }
//                            .addOnFailureListener { e -> Log.e("ML", "Labeling failed", e) }
//
//                        // Object detection
//                        detector.process(img)
//                            .addOnSuccessListener { objs: List<DetectedObject> ->
//                                objectsCount = objs.size
//                            }
//                            .addOnFailureListener { e -> Log.e("ML", "ODT failed", e) }
//                            .addOnCompleteListener { imageProxy.close() }
//                    }
//
//                    val selector = CameraSelector.DEFAULT_BACK_CAMERA
//                    cameraProvider.unbindAll()
//                    cameraProvider.bindToLifecycle(
//                        ctx as androidx.lifecycle.LifecycleOwner,
//                        selector,
//                        preview,
//                        analysis
//                    )
//                }, ContextCompat.getMainExecutor(ctx))
//
//                previewView
//            }
//        )
//
//        // quick, unobtrusive status
//        Text(text = buildString {
//            append("Top label: "); append(topLabel ?: "—")
//            append("   Objects: "); append(objectsCount)
//        },  modifier = Modifier
//            .align(Alignment.TopCenter)
//            .background(Color.Black.copy(alpha = 0.6f))
//            .padding(horizontal = 12.dp, vertical = 6.dp),
//            color = Color.White,
//            fontSize = 16.sp)
//    }
//}
