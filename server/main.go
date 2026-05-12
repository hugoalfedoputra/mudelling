// main.go
package main

import (
	"bytes"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"golang.org/x/crypto/argon2"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

// Database Model
type LogEntry struct {
	ID        string    `gorm:"primaryKey" json:"id"`
	CreatedAt time.Time `json:"timestamp"`
	Key       string    `json:"key"`
	Value     string    `json:"value"`
	Comment   string    `json:"comment"`
}

// Native UUIDv4 Generator
func generateUUID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

// Global vars; reomve the dev_ prefix for production
var fileUploadFolder = "dev_file_uploads"
var dataFolder = "dev_data"
var trainingDB = dataFolder + "/training_logs.db"

// API Key Middleware
func authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		clientKey := r.Header.Get("x-api-key")
		storedHashStr := os.Getenv("API_KEY")

		if storedHashStr == "" {
			http.Error(w, "Server Configuration Error", http.StatusInternalServerError)
			return
		}

		parts := strings.Split(storedHashStr, ".")
		if len(parts) != 2 {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		salt, _ := hex.DecodeString(parts[0])
		expectedHash, _ := hex.DecodeString(parts[1])

		hash := argon2.IDKey([]byte(clientKey), salt, 1, 64*1024, 4, 32)

		if subtle.ConstantTimeCompare(hash, expectedHash) != 1 {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	}
}

// Insert JSON Log (POST)
func handleLogUpload(db *gorm.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Key     string `json:"key"`
			Value   string `json:"value"`
			Comment string `json:"comment"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON body", http.StatusBadRequest)
			return
		}

		entry := LogEntry{
			ID:      generateUUID(),
			Key:     req.Key,
			Value:   req.Value,
			Comment: req.Comment,
		}

		db.Create(&entry)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(entry)
	}
}

// List All Logs (GET)
func handleListLogs(db *gorm.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var logs []LogEntry
		// Retrieve all rows from the database
		if err := db.Find(&logs).Error; err != nil {
			http.Error(w, "Failed to retrieve logs", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(logs)
	}
}

// File Upload (POST)
func handleFileUpload() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		r.ParseMultipartForm(10 << 20) // 10 MB limit

		file, header, err := r.FormFile("file")
		if err != nil {
			http.Error(w, "Failed to get file from form", http.StatusBadRequest)
			return
		}
		defer file.Close()

		// Sanitize filename and extract the actual extension
		safeFilename := filepath.Base(header.Filename)
		ext := strings.ToLower(filepath.Ext(safeFilename))

		// Strict extension allowlist
		if ext != ".txt" && ext != ".csv" && ext != ".npy" {
			http.Error(w, fmt.Sprintf("Unsupported extension '%s'. Only .txt, .csv, .npy allowed", ext), http.StatusBadRequest)
			return
		}

		// Verify contents match the extension
		buf := make([]byte, 512)
		n, _ := file.Read(buf)
		buf = buf[:n] // slice to actual bytes read

		file.Seek(0, io.SeekStart) // Reset file pointer

		switch ext {
		case ".npy":
			if !bytes.HasPrefix(buf, []byte("\x93NUMPY")) {
				http.Error(w, "Unsupported file: Invalid .npy magic bytes", http.StatusBadRequest)
				return
			}
		case ".csv", ".txt":
			contentType := http.DetectContentType(buf)
			if !strings.HasPrefix(contentType, "text/") {
				http.Error(w, "Unsupported file: Does not appear to be text", http.StatusBadRequest)
				return
			}
		}

		// Save the file
		destPath := filepath.Join(fileUploadFolder, safeFilename)
		out, err := os.Create(destPath)
		if err != nil {
			http.Error(w, "Failed to save file", http.StatusInternalServerError)
			return
		}
		defer out.Close()

		io.Copy(out, file)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{
			"message":  "File uploaded successfully",
			"filename": safeFilename,
		})
	}
}

// List All Files (GET)
func handleListFiles() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Read the contents of the directory
		entries, err := os.ReadDir(fileUploadFolder)
		if err != nil {
			http.Error(w, "Failed to read directory", http.StatusInternalServerError)
			return
		}

		var filenames []string
		for _, entry := range entries {
			// Ignore subdirectories, just get actual files
			if !entry.IsDir() {
				filenames = append(filenames, entry.Name())
			}
		}

		// If nil, return an empty array instead of null in JSON
		if filenames == nil {
			filenames = []string{}
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"files": filenames,
			"count": len(filenames),
		})
	}
}

func main() {
	_ = godotenv.Load()

	os.MkdirAll(dataFolder, os.ModePerm)
	os.MkdirAll(fileUploadFolder, os.ModePerm)

	db, err := gorm.Open(sqlite.Open(trainingDB), &gorm.Config{})
	if err != nil {
		panic("Failed to connect to database")
	}
	db.AutoMigrate(&LogEntry{})

	// Routing setup
	http.HandleFunc("/logs", authMiddleware(handleLogUpload(db)))    // POST
	http.HandleFunc("/logs/all", authMiddleware(handleListLogs(db))) // GET
	http.HandleFunc("/upload", authMiddleware(handleFileUpload()))   // POST
	http.HandleFunc("/files", authMiddleware(handleListFiles()))     // GET

	fmt.Println("Server started on port 3333")
	err = http.ListenAndServe(":3333", nil)

	if errors.Is(err, http.ErrServerClosed) {
		fmt.Println("Server closed")
	} else if err != nil {
		fmt.Printf("Error starting server: %s\n", err)
		os.Exit(1)
	}
}
