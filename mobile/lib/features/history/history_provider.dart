import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/services/database_service.dart';

final historyProvider =
    FutureProvider<List<Map<String, dynamic>>>((ref) async {
  return DatabaseService.instance.getRecentHistory(limit: 100);
});
