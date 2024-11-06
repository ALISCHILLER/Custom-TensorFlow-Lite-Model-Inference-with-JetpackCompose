package com.msa.customtensorflowlitemodelinferencewithjetpackcompose

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.compose.ui.tooling.preview.Preview
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import androidx.compose.foundation.Image
import com.msa.customtensorflowlitemodelinferencewithjetpackcompose.ui.theme.CustomTensorFlowLiteModelInferenceWithJetpackComposeTheme
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity1 : ComponentActivity() {

    companion object {
        const val TFLITE_MODEL_NAME = "lite_model_cartoongan.tflite"
        const val IMAGE_SIZE = 224
    }

    private var interpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        createInterpreter()
        setContent {
            CustomTensorFlowLiteModelInferenceWithJetpackComposeTheme {
                MainContent()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter?.close() // آزادسازی منابع Interpreter
        interpreter = null
    }

    private fun createInterpreter() {
        val tfLiteOptions = Interpreter.Options()
        interpreter = getInterpreter(this, TFLITE_MODEL_NAME, tfLiteOptions)
    }

    @Composable
    fun MainContent() {
        var originalBitmap by remember { mutableStateOf<Bitmap?>(null) }
        var processedBitmap by remember { mutableStateOf<Bitmap?>(null) }

        LaunchedEffect(Unit) {
            originalBitmap = loadBitmapFromAsset()
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            originalBitmap?.let { bitmap ->
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Original Image",
                    modifier = Modifier
                        .size(IMAGE_SIZE.dp)
                        .padding(8.dp)
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(onClick = {
                originalBitmap?.let { bitmap ->
                    val inferenceResult = runInference(bitmap)
                    processedBitmap = inferenceResult?.let { convertOutputArrayToImage(it) }
                }
            }) {
                Text(text = "Run Inference")
            }

            Spacer(modifier = Modifier.height(16.dp))

            processedBitmap?.let { bitmap ->
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Processed Image",
                    modifier = Modifier
                        .size(IMAGE_SIZE.dp)
                        .padding(8.dp)
                )
            }
        }
    }

    private fun loadBitmapFromAsset(): Bitmap {
        return applicationContext.assets.open("test.jpg").use { inputStream ->
            BitmapFactory.decodeStream(inputStream).let { bitmap ->
                Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, false)
            }
        }
    }

    private fun runInference(bitmap: Bitmap): Array<Array<Array<FloatArray>>>? {
        if (interpreter == null) return null

        val outputArr = Array(1) { Array(IMAGE_SIZE) { Array(IMAGE_SIZE) { FloatArray(3) } } }
        val byteBuffer = convertBitmapToByteBuffer(bitmap, IMAGE_SIZE, IMAGE_SIZE)
        interpreter?.run(byteBuffer, outputArr)
        return outputArr
    }

    private fun convertOutputArrayToImage(inferenceResult: Array<Array<Array<FloatArray>>>): Bitmap {
        val output = inferenceResult[0]
        val bitmap = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)

        var index = 0
        for (y in 0 until IMAGE_SIZE) {
            for (x in 0 until IMAGE_SIZE) {
                val r = (output[y][x][0] + 1) * 127.5
                val g = (output[y][x][1] + 1) * 127.5
                val b = (output[y][x][2] + 1) * 127.5
                val a = 0xFF
                pixels[index++] = a shl 24 or (r.toInt() shl 16) or (g.toInt() shl 8) or b.toInt()
            }
        }
        bitmap.setPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        return bitmap
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        val mean = 127.5f
        val std = 127.5f

        return ByteBuffer.allocateDirect(1 * width * height * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
            val intValues = IntArray(width * height)
            bitmap.getPixels(intValues, 0, width, 0, 0, width, height)

            for (color in intValues) {
                putFloat((Color.red(color) - mean) / std)
                putFloat((Color.green(color) - mean) / std)
                putFloat((Color.blue(color) - mean) / std)
            }
        }
    }

    private fun getInterpreter(
        context: Context,
        modelName: String,
        tfLiteOptions: Interpreter.Options
    ): Interpreter {
        return Interpreter(FileUtil.loadMappedFile(context, modelName), tfLiteOptions)
    }
}
