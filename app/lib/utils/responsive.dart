import 'package:flutter/material.dart';

enum FormFactor { compact, medium, expanded }

FormFactor formFactorOf(BuildContext context) {
  final w = MediaQuery.sizeOf(context).width;
  if (w < 600) return FormFactor.compact;
  if (w < 900) return FormFactor.medium;
  return FormFactor.expanded;
}

bool isCompact(BuildContext context) =>
    formFactorOf(context) == FormFactor.compact;

bool isMedium(BuildContext context) =>
    formFactorOf(context) == FormFactor.medium;

bool isExpanded(BuildContext context) =>
    formFactorOf(context) == FormFactor.expanded;

/// Whether the current layout should use a two-pane master-detail.
///
/// True on expanded screens (>= 900 dp) or on medium screens in landscape.
bool useTwoPane(BuildContext context) {
  final factor = formFactorOf(context);
  if (factor == FormFactor.expanded) return true;
  if (factor == FormFactor.compact) return false;
  // Medium: two-pane only in landscape.
  return MediaQuery.orientationOf(context) == Orientation.landscape;
}

/// Returns dialog constraints that cap width on small screens.
/// On compact devices the dialog fills the screen width minus 32 dp.
/// On larger devices it is capped at 600 dp.
BoxConstraints dialogConstraints(BuildContext context) {
  final w = MediaQuery.sizeOf(context).width;
  final maxW = w < 600 ? w - 32 : 600.0;
  return BoxConstraints(maxWidth: maxW);
}
