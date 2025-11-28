// src/pages/ChurnDashboard.jsx
import React, { useState } from "react";
import { Pie, Bar } from "react-chartjs-2";
import "bootstrap/dist/css/bootstrap.min.css";
import {
	Chart as ChartJS,
	ArcElement,
	Tooltip,
	Legend,
	CategoryScale,
	LinearScale,
	BarElement,
	Title,
} from "chart.js";

ChartJS.register(
	ArcElement,
	Tooltip,
	Legend,
	CategoryScale,
	LinearScale,
	BarElement,
	Title
);

export default function ChurnDashboard() {
	const [dataRows, setDataRows] = useState([]);
	const [alert, setAlert] = useState("");
	const [charts, setCharts] = useState({});

	const handleSubmit = async (e) => {
		e.preventDefault();
		const formData = new FormData(e.target);
		setAlert("Processing...");

		try {
			const res = await fetch("/predict", {
				method: "POST",
				body: formData,
			});
			const data = await res.json();

			if (!data.length) {
				setAlert("No data returned.");
				return;
			}

			setAlert("Prediction Completed!");
			setDataRows(data);

			// Chart Data
			const churnCounts = { churn: 0, noChurn: 0 };
			const probabilities = [];
			const churnLabels = data.map((row) => parseInt(row.Churn_Label));

			data.forEach((row) => {
				if (row.Churn_Label === 1 || row.Churn_Label === "1")
					churnCounts.churn++;
				else churnCounts.noChurn++;
				probabilities.push(parseFloat(row.Churn_Probability));
			});

			setCharts({
				churnPie: {
					labels: ["Churn", "No Churn"],
					datasets: [
						{
							data: [churnCounts.churn, churnCounts.noChurn],
							backgroundColor: ["#e74c3c", "#2ecc71"],
						},
					],
				},
				probabilityHistogram: {
					labels: probabilities.map((_, i) => `Customer ${i + 1}`),
					datasets: [
						{
							label: "Churn Probability",
							data: probabilities,
							backgroundColor: "#3498db",
						},
					],
				},
				churnLabelPerCustomer: {
					labels: churnLabels.map((_, i) => `Customer ${i + 1}`),
					datasets: [
						{
							label: "Churn Label (0 = No, 1 = Yes)",
							data: churnLabels,
							backgroundColor: churnLabels.map((v) =>
								v === 1 ? "#e74c3c" : "#2ecc71"
							),
						},
					],
				},
			});
		} catch (err) {
			setAlert("Error processing file.");
			console.error(err);
		}
	};

	return (
		<div className="container p-4">
			<h1 className="mb-4 text-center">Customer Churn Prediction</h1>

			<form id="uploadForm" onSubmit={handleSubmit} className="mb-4">
				<div className="input-group">
					<input
						type="file"
						name="file"
						accept=".csv"
						className="form-control"
						required
					/>
					<button type="submit" className="btn btn-primary">
						Predict Churn
					</button>
				</div>
			</form>

			{alert && <div className="alert alert-info">{alert}</div>}

			{/* Charts */}
			{charts.churnPie && (
				<div className="row mb-5">
					<div className="col-md-4">
						<Pie
							data={charts.churnPie}
							options={{
								responsive: true,
								plugins: {
									title: { display: true, text: "Churn Distribution" },
								},
							}}
						/>
					</div>
					<div className="col-md-4">
						<Bar
							data={charts.probabilityHistogram}
							options={{
								responsive: true,
								plugins: {
									title: {
										display: true,
										text: "Churn Probabilities by Customer",
									},
								},
								scales: { y: { beginAtZero: true, max: 1 } },
							}}
						/>
					</div>
					<div className="col-md-4">
						<Bar
							data={charts.churnLabelPerCustomer}
							options={{
								responsive: true,
								plugins: {
									title: { display: true, text: "Churn Label per Customer" },
								},
								scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } },
							}}
						/>
					</div>
				</div>
			)}

			{/* Table */}
			{dataRows.length > 0 && (
				<div className="mt-5">
					<table className="table table-bordered table-striped">
						<thead className="table-dark">
							<tr>
								{Object.keys(dataRows[0]).map((col) => (
									<th key={col}>{col}</th>
								))}
							</tr>
						</thead>
						<tbody>
							{dataRows.map((row, i) => (
								<tr key={i}>
									{Object.values(row).map((val, j) => (
										<td key={j}>{val}</td>
									))}
								</tr>
							))}
						</tbody>
					</table>
				</div>
			)}
		</div>
	);
}
