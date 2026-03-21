// src/api/index.ts
export * from './types';
export { DashboardClient } from './client';

// Default singleton client instance that components can import if desired
export const apiClient = new DashboardClient();
export default apiClient;
