#pragma once

#include <QObject>
#include <QVariantList>
#include <QVariantMap>
#include "../database/database.h"

class BudgetManager : public QObject
{
    Q_OBJECT

public:
    explicit BudgetManager(Database *db, QObject *parent = nullptr);

    Q_INVOKABLE int  addTransaction(int userId, int categoryId, double amount,
                                    bool isExpense, const QString &note, const QString &date);
    Q_INVOKABLE bool updateTransaction(int id, int categoryId, double amount,
                                       bool isExpense, const QString &note, const QString &date);
    Q_INVOKABLE bool deleteTransaction(int id);
    Q_INVOKABLE QVariantList getTransactions(int userId,
                                             const QString &startDate, const QString &endDate);
    Q_INVOKABLE QVariantList getRecentTransactions(int userId, int limit = 10);
    Q_INVOKABLE double       getTotalIncome(int userId, const QString &month);
    Q_INVOKABLE double       getTotalExpenses(int userId, const QString &month);
    Q_INVOKABLE QVariantList getExpensesByCategory(int userId, const QString &month);
    Q_INVOKABLE QVariantList getMonthlyTrend(int userId, int months = 6);

    Q_INVOKABLE bool         setBudget(int userId, int categoryId,
                                       double amount, const QString &month);
    Q_INVOKABLE double       getBudget(int userId, int categoryId, const QString &month);
    Q_INVOKABLE QVariantList getBudgets(int userId, const QString &month);

    Q_INVOKABLE QVariantList getCategories(int userId, bool isExpense);
    Q_INVOKABLE QVariantList getAllCategories(int userId);
    Q_INVOKABLE int  createCategory(const QString &name, const QString &icon,
                                    const QString &color, bool isExpense, int userId);
    Q_INVOKABLE bool deleteCategory(int id);

    Q_INVOKABLE QString currentMonth();
    Q_INVOKABLE QString prevMonth(const QString &month);
    Q_INVOKABLE QString nextMonth(const QString &month);
    Q_INVOKABLE QString formatAmount(double amount);
    Q_INVOKABLE QString monthDisplayName(const QString &month);

signals:
    void dataChanged();

private:
    Database *m_db;
};
