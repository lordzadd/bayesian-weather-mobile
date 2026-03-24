import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/services/prediction_history_dao.dart';

final historyProvider =
    AsyncNotifierProvider<HistoryNotifier, List<PredictionHistoryRow>>(
  HistoryNotifier.new,
);

class HistoryNotifier extends AsyncNotifier<List<PredictionHistoryRow>> {
  final _dao = PredictionHistoryDao();

  @override
  Future<List<PredictionHistoryRow>> build() => _dao.getRecent(limit: 100);

  Future<void> refresh() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(() => _dao.getRecent(limit: 100));
  }
}
