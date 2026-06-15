#pragma once

#include <QObject>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QVariant>
#include <QVariantMap>
#include <QList>
#include <QString>

class Database : public QObject
{
    Q_OBJECT

public:
    explicit Database(QObject *parent = nullptr);
    ~Database();

    bool initialize();
    QSqlDatabase &db() { return m_db; }

    // Users
    int  createUser(const QString &name, const QString &pin,
                    const QString &color, const QString &avatar);
    bool updateUser(int id, const QString &name, const QString &pin,
                    const QString &color, const QString &avatar);
    bool deleteUser(int id);
    QList<QVariantMap> getAllUsers();
    QVariantMap        getUserById(int id);
    bool               verifyPin(int userId, const QString &pin);
    bool               userHasPin(int userId);

    // Categories
    int  createCategory(const QString &name, const QString &icon,
                        const QString &color, bool isExpense, int userId = 0);
    bool updateCategory(int id, const QString &name,
                        const QString &icon, const QString &color);
    bool deleteCategory(int id);
    QList<QVariantMap> getCategories(int userId, bool isExpense);
    QList<QVariantMap> getAllCategories(int userId);

    // Transactions
    int  addTransaction(int userId, int categoryId, double amount,
                        bool isExpense, const QString &note, const QString &date);
    bool updateTransaction(int id, int categoryId, double amount,
                           bool isExpense, const QString &note, const QString &date);
    bool deleteTransaction(int id);
    QList<QVariantMap> getTransactions(int userId,
                                       const QString &startDate, const QString &endDate);
    QList<QVariantMap> getRecentTransactions(int userId, int limit = 10);
    double getTotalIncome(int userId, const QString &month);
    double getTotalExpenses(int userId, const QString &month);
    QList<QVariantMap> getExpensesByCategory(int userId, const QString &month);
    QList<QVariantMap> getMonthlyTrend(int userId, int months = 6);

    // Budgets
    bool   setBudget(int userId, int categoryId, double amount, const QString &month);
    double getBudget(int userId, int categoryId, const QString &month);
    QList<QVariantMap> getBudgets(int userId, const QString &month);

private:
    QSqlDatabase m_db;
    bool createTables();
    void seedDefaultCategories();
};
