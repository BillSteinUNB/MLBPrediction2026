import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath } from 'url'
import path from 'path'
import fs from 'fs'

// ESM-safe root resolution
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const projectRoot = path.resolve(__dirname, '..')

// Custom Vite plugin to serve data files from filesystem
function dataFilesPlugin(): Plugin {
  return {
    name: 'data-files-plugin',
    apply: 'serve',
    configureServer(server) {
      // Add middleware with 'pre' priority to run BEFORE proxy middleware
      server.middlewares.use((req, res, next) => {
        const reqPath = req.url?.split('?')[0] // Strip query params

        // Check if request matches tracker or dual-view patterns
        const trackerMatch = reqPath?.match(/^\/api\/tracker\/(.+)$/)
        const dualViewMatch = reqPath?.match(/^\/api\/dual-view\/(.+)$/)

        if (trackerMatch) {
          const filename = trackerMatch[1]
          const filePath = path.resolve(projectRoot, 'data/reports/run_count/tracker', filename)

          // Security: Ensure file is within allowed directory
          const resolvedPath = path.resolve(filePath)
          const allowedDir = path.resolve(projectRoot, 'data/reports/run_count/tracker')
          if (!resolvedPath.startsWith(allowedDir)) {
            res.statusCode = 403
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: 'Access denied' }))
            return
          }

          // Check if file exists
          if (!fs.existsSync(filePath)) {
            res.statusCode = 404
            res.setHeader('Content-Type', 'application/json')
            res.setHeader('Cache-Control', 'no-cache')
            res.end(JSON.stringify({ error: 'File not found' }))
            return
          }

          // Serve file with appropriate headers
          try {
            const content = fs.readFileSync(filePath, 'utf-8')
            res.statusCode = 200
            res.setHeader('Content-Type', 'application/json')
            res.setHeader('Cache-Control', 'no-cache')
            res.end(content)
          } catch (err) {
            res.statusCode = 500
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: 'Internal server error' }))
          }
          return
        }

        if (dualViewMatch) {
          const filename = dualViewMatch[1]
          const filePath = path.resolve(projectRoot, 'data/reports/run_count/dual_view', filename)

          // Security: Ensure file is within allowed directory
          const resolvedPath = path.resolve(filePath)
          const allowedDir = path.resolve(projectRoot, 'data/reports/run_count/dual_view')
          if (!resolvedPath.startsWith(allowedDir)) {
            res.statusCode = 403
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: 'Access denied' }))
            return
          }

          // Check if file exists
          if (!fs.existsSync(filePath)) {
            res.statusCode = 404
            res.setHeader('Content-Type', 'application/json')
            res.setHeader('Cache-Control', 'no-cache')
            res.end(JSON.stringify({ error: 'File not found' }))
            return
          }

          // Serve file with appropriate headers
          try {
            const content = fs.readFileSync(filePath, 'utf-8')
            res.statusCode = 200
            res.setHeader('Content-Type', 'application/json')
            res.setHeader('Cache-Control', 'no-cache')
            res.end(content)
          } catch (err) {
            res.statusCode = 500
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: 'Internal server error' }))
          }
          return
        }

        const picsMatch = reqPath?.match(/^\/api\/pics\/(.+)$/)

        if (picsMatch) {
          const filename = picsMatch[1]
          const filePath = path.resolve(projectRoot, 'data/reports/pics', filename)

          // Security: Ensure file is within allowed directory
          const resolvedPath = path.resolve(filePath)
          const allowedDir = path.resolve(projectRoot, 'data/reports/pics')
          if (!resolvedPath.startsWith(allowedDir)) {
            res.statusCode = 403
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: 'Access denied' }))
            return
          }

          // Check if file exists
          if (!fs.existsSync(filePath)) {
            res.statusCode = 404
            res.setHeader('Content-Type', 'application/json')
            res.setHeader('Cache-Control', 'no-cache')
            res.end(JSON.stringify({ error: 'File not found' }))
            return
          }

          // Serve file with appropriate headers
          try {
            const content = fs.readFileSync(filePath, 'utf-8')
            res.statusCode = 200
            res.setHeader('Content-Type', 'application/json')
            res.setHeader('Cache-Control', 'no-cache')
            res.end(content)
          } catch (err) {
            res.statusCode = 500
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: 'Internal server error' }))
          }
          return
        }

        next()
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [dataFilesPlugin(), react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8010',
        changeOrigin: true,
      }
    }
  }
})
