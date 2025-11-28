import React, { useState } from "react";
import { Upload, Loader2 } from "lucide-react";
import { processDataset } from "../lib/api";

export default function UploadCard({ onProcessed }) {
	const [file, setFile] = useState(null);
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState("");

	const handleSelect = (e) => {
		setFile(e.target.files?.[0] || null);
		setError("");
	};

	const handleUpload = async () => {
		if (!file) {
			setError("Please select a CSV file.");
			return;
		}
		setBusy(true);
		setError("");
		try {
			const data = await processDataset(file);
			onProcessed?.(data);
		} catch (e) {
			console.error(e);
			setError("Upload failed. Check server and file format.");
		} finally {
			setBusy(false);
		}
	};

	return (
		<section className="bg-white rounded-xl shadow p-6">
			<h2 className="text-lg font-semibold mb-2">Upload Dataset (CSV)</h2>
			<p className="text-sm text-gray-500 mb-4">
				We’ll preprocess (RFM/RFM scaled), cluster with K-Means, render R/F/M
				distributions, and run ANN churn.
			</p>

			<div className="flex flex-col sm:flex-row items-center gap-3">
				<label className="flex-1 h-28 border-2 border-dashed border-gray-300 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:border-indigo-400 transition p-3">
					<Upload className="w-6 h-6 text-gray-400 mb-1" />
					<span className="text-sm text-gray-500">
						Drag & drop or click to select CSV
					</span>
					<input
						type="file"
						accept=".csv"
						className="hidden"
						onChange={handleSelect}
					/>
				</label>

				<div className="w-full sm:w-auto flex-1 sm:flex-none">
					<button
						onClick={handleUpload}
						disabled={busy}
						className="w-full sm:w-auto inline-flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg disabled:opacity-60">
						{busy ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
						{busy ? "Processing…" : "Upload & Process"}
					</button>
					{file && (
						<div className="text-xs text-green-700 mt-2 truncate">
							Selected: {file.name}
						</div>
					)}
					{error && <div className="text-xs text-red-600 mt-2">{error}</div>}
				</div>
			</div>
		</section>
	);
}
