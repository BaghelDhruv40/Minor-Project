// import React, { useState } from "react";
// import UploadCard from "./components/UploadCard";
// import Dashboard from "./pages/Dashboard";

// export default function App() {
// 	const [result, setResult] = useState(null);

// 	return (
// 		<div className="min-h-screen bg-gray-50">
// 			<header className="border-b bg-white">
// 				<div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
// 					<h1 className="text-xl md:text-2xl font-semibold">
// 						Customer Segmentation & Churn Prediction
// 					</h1>
// 					<div className="text-xs text-gray-500">
// 						Flask API:{" "}
// 						{import.meta.env.VITE_API_BASE || "http://localhost:5000"}
// 					</div>
// 				</div>
// 			</header>

// 			<main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
// 				{!result ? (
// 					<UploadCard onProcessed={setResult} />
// 				) : (
// 					<>
// 						<button
// 							onClick={() => setResult(null)}
// 							className="text-sm text-indigo-700 hover:underline">
// 							← Upload another dataset
// 						</button>
// 						<Dashboard result={result} />
// 					</>
// 				)}
// 			</main>
// 		</div>
// 	);
// }

import React, { useState } from "react";
import FileUpload from "./components/FileUpload.jsx";
import PredictionTable from "./components/PredictionTable.jsx";
import { useNavigate } from "react-router-dom";

// import Charts from "./components/Charts.jsx";

function App() {
	const [predictions, setPredictions] = useState([]);
	const navigate = useNavigate();
	return (
		<div className="p-6">
			<h1 className="text-2xl font-bold mb-4">
				Customer Segmentation & Churn Prediction
			</h1>
			<FileUpload onResults={setPredictions} />
			{predictions.length > 0 && (
				<>
					{/* <PredictionTable predictions={predictions} /> */}
					{/* <Charts
						data={[
							{
								label: "Churn",
								count: predictions.filter((p) => p === 1).length,
							},
							{
								label: "Not Churn",
								count: predictions.filter((p) => p === 0).length,
							},
						]}
					/> */}
				</>
			)}
		</div>
	);
}

export default App;
