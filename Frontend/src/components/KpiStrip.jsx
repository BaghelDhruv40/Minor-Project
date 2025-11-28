import React from "react";

const Kpi = ({ label, value, suffix }) => (
	<div className="bg-white rounded-xl shadow p-4">
		<div className="text-xs text-gray-500">{label}</div>
		<div className="text-2xl font-semibold mt-1">
			{value ?? "-"}
			{suffix || ""}
		</div>
	</div>
);

export default function KpiStrip({ summary }) {
	if (!summary) return null;
	const {
		total_customers,
		k,
		avg_recency,
		avg_frequency,
		avg_monetary,
		churn_rate,
	} = summary;

	return (
		<section className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
			<Kpi label="Total Customers" value={total_customers} />
			<Kpi label="Clusters (K)" value={k} />
			<Kpi label="Avg Recency" value={Number(avg_recency)?.toFixed(1)} />
			<Kpi label="Avg Frequency" value={Number(avg_frequency)?.toFixed(1)} />
			<Kpi label="Avg Monetary" value={Number(avg_monetary)?.toFixed(1)} />
			<Kpi
				label="Churn Rate"
				value={(Number(churn_rate) * 100).toFixed(1)}
				suffix="%"
			/>
		</section>
	);
}
