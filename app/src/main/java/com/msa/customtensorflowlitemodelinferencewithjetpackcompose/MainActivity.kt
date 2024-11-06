package com.msa.customtensorflowlitemodelinferencewithjetpackcompose

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.launch
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.msa.customtensorflowlitemodelinferencewithjetpackcompose.ui.theme.CustomTensorFlowLiteModelInferenceWithJetpackComposeTheme
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : ComponentActivity() {

    companion object {
        const val TFLITE_MODEL_NAME = "lite_model_cartoongan.tflite"
    }

    private var interpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CustomTensorFlowLiteModelInferenceWithJetpackComposeTheme {
                MainContent()
            }
        }
    }

    @Composable
    fun MainContent() {
        var originalBitmap by remember { mutableStateOf<Bitmap?>(null) }
        var processedBitmap by remember { mutableStateOf<Bitmap?>(null) }

        // Launcher for taking picture
        val launcher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
            if (bitmap != null) {
                originalBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, false)
            }
        }

        LaunchedEffect(Unit) {
            createInterpreter()
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
                        .size(300.dp)
                        .padding(8.dp)
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Button to take picture
            Button(onClick = {
                launcher.launch()
            }) {
                Text(text = "Take Picture")
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Button to run inference
            Button(onClick = {
                originalBitmap?.let { bitmap ->
                    val inferenceResult = runInference(bitmap)
                    processedBitmap = convertOutputArrayToImage(inferenceResult)
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
                        .size(500.dp)
                        .padding(8.dp)
                )
            }
        }
    }

    private fun convertOutputArrayToImage(inferenceResult: Array<Array<Array<FloatArray>>>): Bitmap {
        val output = inferenceResult[0]
        val bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(224 * 224)

        var index = 0
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val b = (output[y][x][0] + 1) * 127.5
                val r = (output[y][x][1] + 1) * 127.5
                val g = (output[y][x][2] + 1) * 127.5

                val a = 0xFF
                pixels[index] = a shl 24 or (r.toInt() shl 16) or (g.toInt() shl 8) or b.toInt()
                index++
            }
        }
        bitmap.setPixels(pixels, 0, 224, 0, 0, 224, 224)
        return bitmap
    }

    private fun createInterpreter() {
        val tfLiteOptions = Interpreter.Options()
        interpreter = getInterpreter(this, TFLITE_MODEL_NAME, tfLiteOptions)
    }

    private fun runInference(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val outputArr = Array(1) {
            Array(224) {
                Array(224) {
                    FloatArray(3)
                }
            }
        }
        val byteBuffer = convertBitmapToByteBuffer(bitmap, 224, 224)
        interpreter?.run(byteBuffer, outputArr)
        return outputArr
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        val mean = arrayOf(127.5f, 127.5f, 127.5f)
        val standard = arrayOf(127.5f, 127.5f, 127.5f)

        val inputImage = ByteBuffer.allocateDirect(1 * width * height * 3 * 4)
        inputImage.order(ByteOrder.nativeOrder())
        inputImage.rewind()

        val intValues = IntArray(width * height)
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height)

        for (y in 0 until width) {
            for (x in 0 until height) {
                val px = bitmap.getPixel(x, y)
                val r = Color.red(px)
                val g = Color.green(px)
                val b = Color.blue(px)

                val rf = (r - mean[0]) / standard[0]
                val gf = (g - mean[1]) / standard[1]
                val bf = (b - mean[2]) / standard[2]

                inputImage.putFloat(bf)
                inputImage.putFloat(rf)
                inputImage.putFloat(gf)
            }
        }
        return inputImage
    }

    private fun getInterpreter(
        context: Context,
        modelName: String,
        tfLiteOptions: Interpreter.Options
    ): Interpreter {
        return Interpreter(FileUtil.loadMappedFile(context, modelName), tfLiteOptions)
    }
}
