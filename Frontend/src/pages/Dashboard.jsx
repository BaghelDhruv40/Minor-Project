import React from "react";
import KpiStrip from "../components/KpiStrip";
import RfmCharts from "../components/RfmCharts";
import ChurnCharts from "../components/ChurnCharts";
import ChurnTable from "../components/ChurnTable";

export default function Dashboard({ result }) {
	if (!result) return null;

	return (
		<>
			<KpiStrip summary={result.rfm_summary} />

			{(result.charts?.recency_url ||
				result.charts?.frequency_url ||
				result.charts?.monetary_url) && (
				<section className="bg-white rounded-xl shadow p-4 mt-6">
					<div className="flex items-center justify-between mb-4">
						<h2 className="text-lg font-semibold">
							Segmentation — R/F/M Distributions
						</h2>
						<span className="text-xs text-gray-500">
							Matplotlib (from backend)
						</span>
					</div>
					<RfmCharts charts={result.charts} />
				</section>
			)}

			<section className="bg-white rounded-xl shadow p-4 mt-6">
				<div className="flex items-center justify-between mb-4">
					<h2 className="text-lg font-semibold">Churn Prediction</h2>
					<span className="text-xs text-gray-500">ANN results</span>
				</div>
				<ChurnCharts churn={result.churn} />
				<div className="mt-6">
					<ChurnTable rows={result.churn?.rows || []} />
				</div>
			</section>
		</>
	);
}
