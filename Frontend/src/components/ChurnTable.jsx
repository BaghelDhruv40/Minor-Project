import React, { useMemo, useState } from "react";

export default function ChurnTable({ rows }) {
	const [q, setQ] = useState("");
	const [sort, setSort] = useState({ key: "Churn_Probability", dir: "desc" });

	const filtered = useMemo(() => {
		const base = Array.isArray(rows) ? rows : [];
		const search = q.trim().toLowerCase();
		const srt = [...base].filter(
			(r) =>
				!search ||
				String(r.CustomerID ?? "")
					.toLowerCase()
					.includes(search)
		);
		srt.sort((a, b) => {
			const A = a[sort.key] ?? 0;
			const B = b[sort.key] ?? 0;
			return sort.dir === "asc" ? (A > B ? 1 : -1) : A < B ? 1 : -1;
		});
		return srt;
	}, [rows, q, sort]);

	const Th = ({ k, children }) => (
		<th
			className="px-3 py-2 cursor-pointer whitespace-nowrap"
			onClick={() =>
				setSort((s) =>
					s.key === k
						? { key: k, dir: s.dir === "asc" ? "desc" : "asc" }
						: { key: k, dir: "asc" }
				)
			}
			title="Click to sort">
			{children}
			{sort.key === k ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
		</th>
	);

	return (
		<div className="bg-white rounded-lg border">
			<div className="p-3 flex items-center justify-between gap-3">
				<h3 className="font-medium">Customer-Level Predictions</h3>
				<input
					value={q}
					onChange={(e) => setQ(e.target.value)}
					placeholder="Search CustomerID…"
					className="border rounded-md px-3 py-1 text-sm"
				/>
			</div>

			<div className="overflow-x-auto">
				<table className="min-w-full text-sm">
					<thead className="bg-gray-50 border-y">
						<tr>
							<Th k="CustomerID">CustomerID</Th>
							<Th k="Cluster">Cluster</Th>
							<Th k="Recency">Recency</Th>
							<Th k="Frequency">Frequency</Th>
							<Th k="Monetary">Monetary</Th>
							<Th k="Churn_Probability">Churn Probability</Th>
							<Th k="Churn_Label">Churn Label</Th>
						</tr>
					</thead>
					<tbody>
						{filtered.map((r, i) => (
							<tr key={i} className="border-b last:border-b-0 hover:bg-gray-50">
								<td className="px-3 py-2 whitespace-nowrap">{r.CustomerID}</td>
								<td className="px-3 py-2">{r.Cluster}</td>
								<td className="px-3 py-2">{r.Recency}</td>
								<td className="px-3 py-2">{r.Frequency}</td>
								<td className="px-3 py-2">{Number(r.Monetary)?.toFixed(2)}</td>
								<td className="px-3 py-2">
									{Number(r.Churn_Probability)?.toFixed(4)}
								</td>
								<td className="px-3 py-2">
									<span
										className={`px-2 py-0.5 rounded text-xs ${
											Number(r.Churn_Label) === 1
												? "bg-red-100 text-red-700"
												: "bg-green-100 text-green-700"
										}`}>
										{Number(r.Churn_Label) === 1 ? "Churn" : "No Churn"}
									</span>
								</td>
							</tr>
						))}
						{filtered.length === 0 && (
							<tr>
								<td colSpan={7} className="px-3 py-6 text-center text-gray-500">
									No rows
								</td>
							</tr>
						)}
					</tbody>
				</table>
			</div>
		</div>
	);
}
