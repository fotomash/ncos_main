import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import numpy as np


class UnifiedExecutionEngine:
    """Simplified execution engine replacing scattered executors."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def execute(self, strategy_result: Dict[str, Any], analysis_result: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        if strategy_result.get("status") != "match_found":
            return {"status": "no_execution"}

        self.logger.info("Executing %s for %s", strategy_result["strategy"], symbol)

        entry_params = self._calculate_entry_parameters(strategy_result["strategy"], analysis_result, symbol)
        risk_params = self._calculate_risk_parameters(entry_params)

        result = {
            "status": "executed",
            "strategy": strategy_result["strategy"],
            "symbol": symbol,
            "entry_price": entry_params.get("entry_price"),
            "stop_loss": risk_params.get("stop_loss"),
            "take_profit": risk_params.get("take_profit"),
            "position_size": risk_params.get("position_size"),
            "direction": entry_params.get("direction"),
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._log_execution(result)
        return result

    def _calculate_entry_parameters(self, strategy: str, analysis: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        base_price = 1.1000 if "EUR" in symbol else 2000.0
        return {
            "entry_price": base_price * (1 + np.random.normal(0, 0.001)),
            "direction": "buy" if np.random.random() > 0.5 else "sell",
            "entry_reason": f"{strategy} setup confirmed",
        }

    def _calculate_risk_parameters(self, entry_params: Dict[str, Any]) -> Dict[str, Any]:
        entry_price = entry_params.get("entry_price", 1.0)
        direction = entry_params.get("direction", "buy")

        risk_pct = self.config.get("risk_percent", 1.0) / 100
        sl_buffer = entry_price * 0.01
        tp_ratio = self.config.get("tp_rr", 3.0)

        if direction == "buy":
            stop_loss = entry_price - sl_buffer
            take_profit = entry_price + (sl_buffer * tp_ratio)
        else:
            stop_loss = entry_price + sl_buffer
            take_profit = entry_price - (sl_buffer * tp_ratio)

        risk_amount = self.config.get("account_size", 10000) * risk_pct
        position_size = risk_amount / sl_buffer

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": round(position_size, 2),
            "risk_amount": risk_amount,
        }

    def _log_execution(self, execution_result: Dict[str, Any]) -> None:
        log_dir = Path("journal")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"execution_{execution_result['symbol']}_{timestamp}.json"

        try:
            with open(log_file, "w") as f:
                json.dump(execution_result, f, indent=2)
        except Exception as exc:  # pragma: no cover - logging should not fail tests
            self.logger.error("Failed to log execution: %s", exc)
