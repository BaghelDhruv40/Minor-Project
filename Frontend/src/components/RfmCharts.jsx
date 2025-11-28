import React from "react";

/** Shows Matplotlib images coming from the backend. */
export default function RfmCharts({ charts }) {
	if (!charts) return null;
	const { recency_url, frequency_url, monetary_url } = charts;

	const Card = ({ title, src }) => (
		<div className="bg-gray-50 rounded-lg border p-3">
			<div className="text-sm font-medium mb-2">{title}</div>
			{src ? (
				<img
					src={src}
					alt={title}
					className="w-full h-auto rounded"
					loading="lazy"
				/>
			) : (
				<div className="text-xs text-gray-400">No chart provided</div>
			)}
		</div>
	);

	return (
		<div className="grid md:grid-cols-3 gap-4">
			<Card title="Recency Distribution by Cluster" src={recency_url} />
			<Card title="Frequency Distribution by Cluster" src={frequency_url} />
			<Card title="Monetary Distribution by Cluster" src={monetary_url} />
		</div>
	);
}
