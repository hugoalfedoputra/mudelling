package main

import (
	"crypto/rand"
	"fmt"
	"os"

	"golang.org/x/crypto/argon2"
)

func main() {
	plaintextKey := os.Args[1]

	// Generate 16 bytes of random salt
	salt := make([]byte, 16)
	if _, err := rand.Read(salt); err != nil {
		panic(err)
	}

	// Hash using Argon2id
	// Parameters: password, salt, iterations, memory (KB), threads, key length
	hash := argon2.IDKey([]byte(plaintextKey), salt, 1, 64*1024, 4, 32)

	// Format as hex for easy storage in .env
	encodedHash := fmt.Sprintf("%x.%x", salt, hash)

	fmt.Println("Store this API key properly lest you reconfigure your entire stack:")
	fmt.Println("API_KEY_PLAINTEXT:", plaintextKey)
	fmt.Println("\nPut this in your .env file on the Go server:")
	fmt.Println("API_KEY=" + encodedHash)
}
