#ifndef AESIMULTIPRECISION_BENCHMARKS_H
#define AESIMULTIPRECISION_BENCHMARKS_H

#include <iostream>
#include <filesystem>
#include <sqlite3.h>

namespace Logging {
    class DatabaseInterface final {
        sqlite3* db = nullptr;

        class DatabaseMessage final {
            char* message = nullptr;
        public:
            DatabaseMessage() = default;
            operator char* () { return message; }
            operator char** () { return &message; }
            ~DatabaseMessage() { sqlite3_free(message); }
        };

    public:
        DatabaseInterface() = delete;

        DatabaseInterface(const std::filesystem::path& dbLocation) {
            if(sqlite3_open(dbLocation.c_str(), &db) != SQLITE_OK)
                throw std::runtime_error("Failed to open database");
        }

        void exec(const std::string& query) {
            DatabaseMessage message {};
            if (sqlite3_exec(db, query.c_str(), nullptr, nullptr, message) != SQLITE_OK)
                throw std::runtime_error(std::string("Query failed: ") + message.operator char *());
        }

        ~DatabaseInterface() {
            if(sqlite3_close(db) != SQLITE_OK)
                std::cerr << "Failed to close database connection." << std::endl;
        }
    };

    static std::filesystem::path getTimeDbLocation() {
#define TimeDatabase "measures.db"
        return std::filesystem::path(__FILE__).parent_path() / TimeDatabase;
    }

    static void addRecord(const std::string &table, std::time_t date, std::size_t timeMs) {
        using namespace std::string_literals;

        std::tm *timeInfo = std::localtime(&date);
        char buffer[20]{};
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeInfo);

        try {
            DatabaseInterface db(getTimeDbLocation());
            db.exec("CREATE TABLE IF NOT EXISTS "s + table +
                    "(test_date DATETIME PRIMARY KEY NOT NULL, time_ms INTEGER NOT NULL);");
            db.exec("INSERT INTO "s + table + "(test_date, time_ms) VALUES ('" + buffer + "', " +
                    std::to_string(timeMs) + ");");
        } catch (const std::exception &e) {
            std::cout << "Add time measurement for " << table << " failed: " << e.what() << "." << std::endl;
        }
    }
}

#endif //AESIMULTIPRECISION_BENCHMARKS_H
