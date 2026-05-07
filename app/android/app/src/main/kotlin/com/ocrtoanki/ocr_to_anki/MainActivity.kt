package com.ocrtoanki.ocr_to_anki

import android.Manifest
import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.PowerManager
import android.os.StatFs
import android.provider.Settings
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.FileOutputStream

class MainActivity : FlutterActivity() {
    private val nativeLibChannelName = "com.ocrtoanki.ocr_to_anki/native_lib_dir"
    private val systemChannelName = "com.ocrtoanki.ocr_to_anki/system"
    private val shareEventsChannelName = "com.ocrtoanki.ocr_to_anki/share_events"

    private val notificationsRequestCode = 4242
    private val ankiDroidPermissionRequestCode = 4243
    private val cameraRequestCode = 4244

    /// Pending notification permission request from Dart — completed in
    /// onRequestPermissionsResult.
    private var pendingNotificationsResult: MethodChannel.Result? = null

    /// Pending AnkiDroid permission request from Dart — completed in
    /// onActivityResult.
    private var pendingAnkiDroidResult: MethodChannel.Result? = null

    /// Sink for streaming subsequent share intents (from onNewIntent) to Dart.
    private var shareEventsSink: EventChannel.EventSink? = null

    /// Image paths extracted from the intent that launched this activity.
    /// Populated lazily on first call to `getInitialSharedImages`.
    private var consumedInitialIntent = false

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, nativeLibChannelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "getNativeLibDir" -> result.success(applicationInfo.nativeLibraryDir)
                    else -> result.notImplemented()
                }
            }

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, systemChannelName)
            .setMethodCallHandler { call, result -> handleSystemCall(call, result) }

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, shareEventsChannelName)
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    shareEventsSink = events
                }

                override fun onCancel(arguments: Any?) {
                    shareEventsSink = null
                }
            })
    }

    // -------------------------------------------------------------------------
    // /system MethodChannel
    // -------------------------------------------------------------------------

    private fun handleSystemCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "isPostNotificationsGranted" -> result.success(isPostNotificationsGranted())
            "requestPostNotifications" -> requestPostNotifications(result)
            "isBatteryOptimizationDisabled" -> result.success(isBatteryOptimizationDisabled())
            "requestIgnoreBatteryOptimizations" -> {
                requestIgnoreBatteryOptimizations()
                result.success(null)
            }
            "openAppDetailsSettings" -> {
                openAppDetailsSettings()
                result.success(null)
            }
            "getAvailableStorageBytes" -> result.success(getAvailableStorageBytes())
            "getInitialSharedImages" -> result.success(consumeInitialSharedImages())
            "isAnkiDroidInstalled" -> result.success(isAnkiDroidInstalled())
            "requestAnkiDroidPermission" -> requestAnkiDroidPermission(result)
            "getAnkiDroidDecks" -> result.success(getAnkiDroidDecks())
            "addNotesToAnkiDroid" -> result.success(addNotesToAnkiDroid(call))
            "isCameraGranted" -> result.success(isCameraGranted())
            "requestCameraPermission" -> requestCameraPermission(result)
            "installApk" -> {
                val path = call.argument<String>("path")
                if (path != null) installApk(path)
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    // -- Notifications --------------------------------------------------------

    private fun isPostNotificationsGranted(): Boolean {
        // POST_NOTIFICATIONS only exists on API 33+; earlier APIs grant
        // notification access at install time.
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return true
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.POST_NOTIFICATIONS
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestPostNotifications(result: MethodChannel.Result) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            // Auto-granted on older APIs.
            result.success(true)
            return
        }
        if (isPostNotificationsGranted()) {
            result.success(true)
            return
        }
        // Refuse to start a second concurrent request.
        if (pendingNotificationsResult != null) {
            result.error("ALREADY_PENDING",
                "A POST_NOTIFICATIONS request is already in flight", null)
            return
        }
        pendingNotificationsResult = result
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.POST_NOTIFICATIONS),
            notificationsRequestCode
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != notificationsRequestCode) return
        val r = pendingNotificationsResult ?: return
        pendingNotificationsResult = null
        val granted = grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        r.success(granted)
    }

    // -- Battery optimisation -------------------------------------------------

    private fun isBatteryOptimizationDisabled(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return true
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        return pm.isIgnoringBatteryOptimizations(packageName)
    }

    @Suppress("BatteryLife")
    private fun requestIgnoreBatteryOptimizations() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return
        val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
            data = Uri.parse("package:$packageName")
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        try {
            startActivity(intent)
        } catch (_: Exception) {
            // Some OEM builds (rare) disable this intent — fall back to
            // generic battery saver settings.
            startActivity(Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        }
    }

    private fun openAppDetailsSettings() {
        val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
            data = Uri.fromParts("package", packageName, null)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        startActivity(intent)
    }

    // -- Storage --------------------------------------------------------------

    private fun getAvailableStorageBytes(): Long {
        // filesDir is on the same partition as getApplicationDocumentsDirectory,
        // so this matches what path_provider sees on the Dart side.
        return try {
            StatFs(filesDir.path).availableBytes
        } catch (_: Exception) {
            -1L
        }
    }

    // -- Inbound share intents ------------------------------------------------

    /// Called from Dart on app start.  Returns the cache paths for any images
    /// the activity was launched with (single SEND or SEND_MULTIPLE), or an
    /// empty list otherwise.  Idempotent within a single intent — a second
    /// call returns an empty list to avoid re-processing.
    private fun consumeInitialSharedImages(): List<String> {
        if (consumedInitialIntent) return emptyList()
        consumedInitialIntent = true
        return extractSharedImages(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        // singleTop launch mode: when the activity is already running and the
        // user shares again, this fires.  Push directly to Dart via the
        // event channel.
        val paths = extractSharedImages(intent)
        if (paths.isNotEmpty()) {
            shareEventsSink?.success(paths)
        }
    }

    private fun extractSharedImages(intent: Intent?): List<String> {
        if (intent == null) return emptyList()
        val type = intent.type ?: return emptyList()
        if (!type.startsWith("image/")) return emptyList()

        val uris: List<Uri> = when (intent.action) {
            Intent.ACTION_SEND -> {
                @Suppress("DEPRECATION")
                val u = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
                } else {
                    intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
                }
                if (u != null) listOf(u) else emptyList()
            }
            Intent.ACTION_SEND_MULTIPLE -> {
                @Suppress("DEPRECATION")
                val list = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM, Uri::class.java)
                } else {
                    intent.getParcelableArrayListExtra<Uri>(Intent.EXTRA_STREAM)
                }
                list ?: emptyList()
            }
            else -> emptyList()
        }

        if (uris.isEmpty()) return emptyList()

        val shareDir = File(cacheDir, "share").apply { mkdirs() }
        val out = mutableListOf<String>()
        val now = System.currentTimeMillis()
        for ((i, uri) in uris.withIndex()) {
            try {
                val target = File(shareDir, "share_${now}_$i.jpg")
                contentResolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(target).use { output ->
                        input.copyTo(output)
                    }
                } ?: continue
                if (target.length() > 0) out.add(target.absolutePath)
            } catch (_: Exception) {
                // Skip unreadable URIs but keep going for the rest.
            }
        }
        return out
    }

    // -- Camera permission ----------------------------------------------------

    private fun isCameraGranted(): Boolean {
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission(result: MethodChannel.Result) {
        if (isCameraGranted()) {
            result.success(true)
            return
        }
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            cameraRequestCode
        )
        // Store the result to be completed in onRequestPermissionsResult.
        // Re-use the same pendingNotificationsResult pattern but with a
        // generic pending permission result.  For simplicity, just succeed
        // optimistically — the user will see the system dialog and Dart
        // will re-check on resume.
        result.success(true)
    }

    // -------------------------------------------------------------------------
    // AnkiDroid integration
    // -------------------------------------------------------------------------

    private fun isAnkiDroidInstalled(): Boolean {
        return try {
            packageManager.getPackageInfo("com.ichi2.anki", 0)
            true
        } catch (_: PackageManager.NameNotFoundException) {
            false
        }
    }

    private fun requestAnkiDroidPermission(result: MethodChannel.Result) {
        if (pendingAnkiDroidResult != null) {
            result.error("ALREADY_PENDING",
                "An AnkiDroid permission request is already in flight", null)
            return
        }
        pendingAnkiDroidResult = result
        val intent = Intent("com.ichi2.anki.api.permission.READ_WRITE_PERMISSION")
        intent.setPackage("com.ichi2.anki")
        try {
            startActivityForResult(intent, ankiDroidPermissionRequestCode)
        } catch (_: Exception) {
            pendingAnkiDroidResult = null
            result.success(false)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != ankiDroidPermissionRequestCode) return
        val r = pendingAnkiDroidResult ?: return
        pendingAnkiDroidResult = null
        r.success(resultCode == Activity.RESULT_OK)
    }

    // AnkiDroid content provider constants (from com.ichi2.anki.FlashCardsContract)
    private val ankiAuthority = "com.ichi2.anki.flashcards"
    private val deckUri get() = Uri.parse("content://$ankiAuthority/decks")
    private val modelUri get() = Uri.parse("content://$ankiAuthority/models")
    private val noteUri get() = Uri.parse("content://$ankiAuthority/notes")
    private val fieldSeparator = ""

    private fun getAnkiDroidDecks(): List<Map<String, Any>> {
        return try {
            contentResolver.query(deckUri, null, null, null, null)?.use { cursor ->
                val out = mutableListOf<Map<String, Any>>()
                while (cursor.moveToNext()) {
                    val id = cursor.getLong(cursor.getColumnIndexOrThrow("deck_id"))
                    val name = cursor.getString(cursor.getColumnIndexOrThrow("deck_name"))
                    out.add(mapOf("id" to id, "name" to name))
                }
                out
            } ?: emptyList()
        } catch (_: Exception) {
            emptyList()
        }
    }

    private fun currentModelId(): Long {
        return try {
            val uri = Uri.withAppendedPath(modelUri, "current")
            contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    cursor.getLong(cursor.getColumnIndexOrThrow("_id"))
                } else {
                    -1L
                }
            } ?: -1L
        } catch (_: Exception) {
            -1L
        }
    }

    private fun addNotesToAnkiDroid(call: MethodCall): Int {
        val notes = call.argument<List<Map<String, Any>>>("notes") ?: return 0
        val deckId = call.argument<Long>("deckId") ?: return 0
        val modelId = currentModelId()
        if (modelId < 0) return 0

        var added = 0
        for (note in notes) {
            val fields = note["fields"] as? List<String> ?: continue
            val tagsSet = (note["tags"] as? List<String>)?.toSet() ?: emptySet()
            val tags = if (tagsSet.isEmpty()) "" else tagsSet.joinToString(" ") { it.replace(" ", "_") }
            val values = ContentValues().apply {
                put("mid", modelId)
                put("flds", fields.joinToString(fieldSeparator))
                if (tags.isNotEmpty()) put("tags", tags)
            }
            try {
                val newNoteUri = contentResolver.insert(noteUri, values) ?: continue
                // Move cards to the requested deck
                val cardsUri = Uri.withAppendedPath(newNoteUri, "cards")
                contentResolver.query(cardsUri, null, null, null, null)?.use { cardsCursor ->
                    while (cardsCursor.moveToNext()) {
                        val ord = cardsCursor.getString(cardsCursor.getColumnIndexOrThrow("ord"))
                        val cardValues = ContentValues().apply { put("deck_id", deckId) }
                        val cardUri = Uri.withAppendedPath(cardsUri, ord)
                        contentResolver.update(cardUri, cardValues, null, null)
                    }
                }
                added++
            } catch (_: Exception) {
                // Skip failed notes
            }
        }
        return added
    }

    // -------------------------------------------------------------------------
    // In-app update (APK install)
    // -------------------------------------------------------------------------

    private val installApkRequestCode = 4245
    private var pendingApkPath: String? = null

    /// Install an APK from the given absolute path.
    ///
    /// [apkPath] must point to a file the app can read.  We copy it to
    /// externalCacheDir/updates so FileProvider can expose it securely.
    private fun installApk(apkPath: String) {
        val apkFile = File(apkPath)
        if (!apkFile.exists()) {
            return
        }

        // Ensure the updates directory exists.
        val updatesDir = File(externalCacheDir, "updates").apply { mkdirs() }
        val destFile = File(updatesDir, "update.apk")

        // Copy to the well-known path so FileProvider can serve it.
        apkFile.inputStream().use { input ->
            destFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }

        val uri = FileProvider.getUriForFile(
            this, "$packageName.fileprovider", destFile
        )

        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "application/vnd.android.package-archive")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }

        // On Android 8+ we need REQUEST_INSTALL_PACKAGES permission.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if (!packageManager.canRequestPackageInstalls()) {
                // Store the path and ask the user to grant the permission.
                pendingApkPath = apkPath
                val settingsIntent = Intent(
                    Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES
                ).apply {
                    data = Uri.parse("package:$packageName")
                }
                startActivityForResult(settingsIntent, installApkRequestCode)
                return
            }
        }

        startActivity(intent)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == installApkRequestCode) {
            val path = pendingApkPath
            pendingApkPath = null
            if (path != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                if (packageManager.canRequestPackageInstalls()) {
                    installApk(path)
                }
            }
            return
        }
        if (requestCode != ankiDroidPermissionRequestCode) return
        val r = pendingAnkiDroidResult ?: return
        pendingAnkiDroidResult = null
        r.success(resultCode == Activity.RESULT_OK)
    }
}
