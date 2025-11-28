import React, { useMemo } from "react";
import {
	PieChart,
	Pie,
	Cell,
	Tooltip as RTooltip,
	Legend,
	BarChart,
	Bar,
	XAxis,
	YAxis,
	CartesianGrid,
} from "recharts";

export default function ChurnCharts({ churn }) {
	const rows = churn?.rows || [];
	const counts = churn?.counts || { churn: 0, noChurn: 0 };

	const pieData = useMemo(
		() => [
			{ name: "Churn", value: counts.churn || 0 },
			{ name: "No Churn", value: counts.noChurn || 0 },
		],
		[counts]
	);

	const probData = useMemo(
		() =>
			rows.map((r, i) => ({
				idx: i + 1,
				probability: Number(r.Churn_Probability ?? 0),
			})),
		[rows]
	);

	const labelData = useMemo(
		() =>
			rows.map((r, i) => ({
				idx: i + 1,
				label: Number(r.Churn_Label ?? 0),
			})),
		[rows]
	);

	return (
		<div className="grid lg:grid-cols-3 gap-6">
			{/* Pie: Churn vs No Churn */}
			<div className="bg-gray-50 rounded-lg border p-3">
				<div className="text-sm font-medium mb-2">Churn Distribution</div>
				<PieChart width={320} height={240}>
					<Pie
						data={pieData}
						dataKey="value"
						nameKey="name"
						cx="50%"
						cy="50%"
						outerRadius={80}>
						{pieData.map((_, i) => (
							<Cell key={i} />
						))}
					</Pie>
					<RTooltip />
					<Legend />
				</PieChart>
			</div>

			{/* Bar: Churn probability per customer */}
			<div className="bg-gray-50 rounded-lg border p-3">
				<div className="text-sm font-medium mb-2">
					Churn Probabilities by Customer
				</div>
				<BarChart width={360} height={240} data={probData}>
					<CartesianGrid strokeDasharray="3 3" />
					<XAxis dataKey="idx" tick={false} />
					<YAxis domain={[0, 1]} />
					<RTooltip />
					<Bar dataKey="probability" />
				</BarChart>
			</div>

			{/* Bar: Churn label per customer (0/1) */}
			<div className="bg-gray-50 rounded-lg border p-3">
				<div className="text-sm font-medium mb-2">Churn Label per Customer</div>
				<BarChart width={360} height={240} data={labelData}>
					<CartesianGrid strokeDasharray="3 3" />
					<XAxis dataKey="idx" tick={false} />
					<YAxis ticks={[0, 1]} domain={[0, 1]} />
					<RTooltip />
					<Bar dataKey="label" />
				</BarChart>
			</div>
		</div>
	);
}
