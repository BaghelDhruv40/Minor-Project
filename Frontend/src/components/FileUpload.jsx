import React, { useState } from "react";
import { Upload, Loader2 } from "lucide-react";

export default function FileUpload() {
	const [file, setFile] = useState(null);
	const [isUploading, setIsUploading] = useState(false);

	const handleFileChange = (e) => {
		setFile(e.target.files[0]);
	};

	const handleUpload = async () => {
		if (!file) return;
		setIsUploading(true);

		const formData = new FormData();
		formData.append("file", file);

		try {
			const res = await fetch("http://localhost:5000/upload", {
				method: "POST",
				body: formData,
			});
			const data = await res.json();
			console.log(data);
		} catch (error) {
			console.error(error);
		} finally {
			setIsUploading(false);
		}
	};

	return (
		<div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
			<div className="bg-white rounded-2xl shadow-lg p-8 w-full max-w-lg">
				<h1 className="text-2xl font-bold text-gray-800 mb-4 text-center">
					Customer Segmentation & Churn Prediction
				</h1>
				<p className="text-gray-500 text-sm mb-6 text-center">
					Upload a CSV file containing your customer data.
				</p>

				{/* File input */}
				<label className="flex flex-col items-center justify-center w-full h-40 border-2 border-dashed border-gray-300 rounded-xl cursor-pointer hover:border-indigo-400 transition">
					<Upload className="w-10 h-10 text-gray-400 mb-2" />
					<span className="text-gray-500 text-sm">
						Drag & drop or click to select file
					</span>
					<input
						type="file"
						accept=".csv"
						className="hidden"
						onChange={handleFileChange}
					/>
				</label>

				{file && (
					<p className="mt-3 text-sm text-green-600 text-center">
						Selected: {file.name}
					</p>
				)}

				{/* Upload Button */}
				<button
					onClick={handleUpload}
					disabled={isUploading || !file}
					className="mt-6 w-full bg-indigo-500 hover:bg-indigo-600 text-white font-medium py-2 px-4 rounded-lg flex items-center justify-center disabled:opacity-50">
					{isUploading ? (
						<>
							<Loader2 className="w-5 h-5 animate-spin mr-2" /> Uploading...
						</>
					) : (
						"Upload & Process"
					)}
				</button>
			</div>
		</div>
	);
}

// import React, { useState } from "react";
// import axios from "axios";

// export default function FileUpload({ onResults }) {
// 	const [file, setFile] = useState(null);

// 	const handleUpload = async () => {
// 		const formData = new FormData();
// 		formData.append("file", file);
// 		const res = await axios.post("http://localhost:5000/predict", formData);
// 		onResults(res.data.predictions);
// 	};

// 	return (
// 		<div className="p-4 border rounded-lg">
// 			<input
// 				type="file"
// 				onChange={(e) => setFile(e.target.files[0])}
// 				className="mb-2"
// 			/>
// 			<button
// 				onClick={handleUpload}
// 				className="bg-blue-500 text-white px-4 py-2 rounded">
// 				Upload & Predict
// 			</button>
// 		</div>
// 	);
// }
