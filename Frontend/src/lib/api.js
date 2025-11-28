import axios from "axios";

const api = axios.create({
	baseURL: import.meta.env.VITE_API_BASE || "http://localhost:5000",
});

export const processDataset = async (file) => {
	const form = new FormData();
	form.append("file", file);
	const { data } = await api.post("/api/process", form, {
		headers: { "Content-Type": "multipart/form-data" },
	});
	return data;
};

export default api;
