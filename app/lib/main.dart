import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'screens/screens.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ProviderScope(child: OcrToAnkiApp()));
}

class OcrToAnkiApp extends StatelessWidget {
  const OcrToAnkiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'OCR to Anki',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.dark,
      theme: ThemeData(
        colorSchemeSeed: Colors.deepOrange,
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: Colors.deepOrange,
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      initialRoute: '/',
      routes: {
        '/': (_) => const HomeScreen(),
        '/processing': (_) => const ProcessingScreen(),
        '/review': (_) => const ReviewScreen(),
        '/settings': (_) => const SettingsScreen(),
        '/history': (_) => const HistoryScreen(),
      },
    );
  }
}
