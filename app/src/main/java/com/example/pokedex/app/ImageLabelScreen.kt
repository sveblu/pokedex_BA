package com.example.pokedex.app

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.layout.ContentScale
import coil.compose.AsyncImage
import com.example.pokedex.ml.ImageLabeler
import com.example.pokedex.ml.LabelResult
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ImageLabelScreen() {
    val ctx = LocalContext.current
    val labeler = remember { ImageLabeler() }
    val scope = rememberCoroutineScope()

    var picked: Uri? by remember { mutableStateOf(null) }
    var results by remember { mutableStateOf<List<LabelResult>>(emptyList()) }
    var running by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }

    val picker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        picked = uri
        results = emptyList()
        error = null
        if (uri != null) {
            running = true
            scope.launch {
                try {
                    results = labeler.labelFromUri(ctx, uri)
                } catch (e: Exception) {
                    error = e.localizedMessage ?: "Labeling failed"
                } finally {
                    running = false
                }
            }
        }
    }

    Scaffold(
        topBar = { TopAppBar(title = { Text("Image Labeling") }) },
        floatingActionButton = {
            FloatingActionButton(onClick = {
                picker.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
            }) { Text("Pick") }
        }
    ) { pad ->
        Column(
            Modifier
                .fillMaxSize()
                .padding(pad)
                .padding(12.dp)
        ) {
            if (picked != null) {
                AsyncImage(
                    model = picked,                // Uri? is fine here
                    contentDescription = null,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 220.dp),
                    contentScale = ContentScale.Fit
                )
            } else {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .height(220.dp),
                    contentAlignment = Alignment.Center
                ) { Text("Tap Pick to choose a photo") }
            }

            Spacer(Modifier.height(12.dp))

            when {
                running -> LinearProgressIndicator(Modifier.fillMaxWidth())
                error != null -> Text("Error: $error", color = MaterialTheme.colorScheme.error)
                results.isNotEmpty() -> {
                    Text("Top labels", fontWeight = FontWeight.SemiBold)
                    LazyColumn {
                        items(results.take(5)) { r ->
                            Row(
                                Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(r.text)
                                Text("${(r.confidence * 100).toInt()}%")
                            }
                        }
                    }
                }
            }
        }
    }
}
