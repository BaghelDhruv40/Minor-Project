// components/PredictionTable.js
import React from "react";

export default function PredictionTable({ predictions }) {
	return (
		<table className="table-auto border-collapse border border-gray-300 w-full">
			<thead>
				<tr>
					<th className="border px-4 py-2">Customer ID</th>
					<th className="border px-4 py-2">Prediction</th>
				</tr>
			</thead>
			<tbody>
				{predictions.map((pred, i) => (
					<tr key={i}>
						<td className="border px-4 py-2">{i + 1}</td>
						<td className="border px-4 py-2">{pred}</td>
					</tr>
				))}
			</tbody>
		</table>
	);
}
