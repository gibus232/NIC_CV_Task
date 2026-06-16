#include "budgetmanager.h"
#include <QDateTime>
#include <QLocale>

BudgetManager::BudgetManager(Database *db, QObject *parent)
    : QObject(parent), m_db(db) {}

int BudgetManager::addTransaction(int userId, int categoryId, double amount,
                                   bool isExpense, const QString &note, const QString &date)
{
    int id = m_db->addTransaction(userId, categoryId, amount, isExpense, note, date);
    if (id > 0) emit dataChanged();
    return id;
}

bool BudgetManager::updateTransaction(int id, int categoryId, double amount,
                                      bool isExpense, const QString &note, const QString &date)
{
    bool ok = m_db->updateTransaction(id, categoryId, amount, isExpense, note, date);
    if (ok) emit dataChanged();
    return ok;
}

bool BudgetManager::deleteTransaction(int id)
{
    bool ok = m_db->deleteTransaction(id);
    if (ok) emit dataChanged();
    return ok;
}

QVariantList BudgetManager::getTransactions(int userId,
                                            const QString &startDate, const QString &endDate)
{
    QVariantList r;
    for (auto &m : m_db->getTransactions(userId, startDate, endDate)) r << m;
    return r;
}

QVariantList BudgetManager::getRecentTransactions(int userId, int limit)
{
    QVariantList r;
    for (auto &m : m_db->getRecentTransactions(userId, limit)) r << m;
    return r;
}

double BudgetManager::getTotalIncome(int userId, const QString &month)
    { return m_db->getTotalIncome(userId, month); }

double BudgetManager::getTotalExpenses(int userId, const QString &month)
    { return m_db->getTotalExpenses(userId, month); }

QVariantList BudgetManager::getExpensesByCategory(int userId, const QString &month)
{
    QVariantList r;
    for (auto &m : m_db->getExpensesByCategory(userId, month)) r << m;
    return r;
}

QVariantList BudgetManager::getMonthlyTrend(int userId, int months)
{
    QVariantList r;
    for (auto &m : m_db->getMonthlyTrend(userId, months)) r << m;
    return r;
}

bool BudgetManager::setBudget(int userId, int categoryId,
                              double amount, const QString &month)
{
    bool ok = m_db->setBudget(userId, categoryId, amount, month);
    if (ok) emit dataChanged();
    return ok;
}

double BudgetManager::getBudget(int userId, int categoryId, const QString &month)
    { return m_db->getBudget(userId, categoryId, month); }

QVariantList BudgetManager::getBudgets(int userId, const QString &month)
{
    QVariantList r;
    for (auto &m : m_db->getBudgets(userId, month)) r << m;
    return r;
}

QVariantList BudgetManager::getCategories(int userId, bool isExpense)
{
    QVariantList r;
    for (auto &m : m_db->getCategories(userId, isExpense)) r << m;
    return r;
}

QVariantList BudgetManager::getAllCategories(int userId)
{
    QVariantList r;
    for (auto &m : m_db->getAllCategories(userId)) r << m;
    return r;
}

int BudgetManager::createCategory(const QString &name, const QString &icon,
                                   const QString &color, bool isExpense, int userId)
{
    int id = m_db->createCategory(name, icon, color, isExpense, userId);
    if (id > 0) emit dataChanged();
    return id;
}

bool BudgetManager::deleteCategory(int id)
{
    bool ok = m_db->deleteCategory(id);
    if (ok) emit dataChanged();
    return ok;
}

QString BudgetManager::currentMonth()
    { return QDateTime::currentDateTime().toString("yyyy-MM"); }

QString BudgetManager::prevMonth(const QString &month)
{
    auto dt = QDateTime::fromString(month + "-01", "yyyy-MM-dd");
    return dt.addMonths(-1).toString("yyyy-MM");
}

QString BudgetManager::nextMonth(const QString &month)
{
    auto dt = QDateTime::fromString(month + "-01", "yyyy-MM-dd");
    return dt.addMonths(1).toString("yyyy-MM");
}

QString BudgetManager::formatAmount(double amount)
{
    return QString("%1 ₽").arg(QLocale(QLocale::Russian).toString(amount, 'f', 2));
}

QString BudgetManager::monthDisplayName(const QString &month)
{
    static const QStringList names = {
        "Январь",
        "Февраль",
        "Март",
        "Апрель",
        "Май",
        "Июнь",
        "Июль",
        "Август",
        "Сентябрь",
        "Октябрь",
        "Ноябрь",
        "Декабрь"
    };
    auto dt = QDateTime::fromString(month + "-01", "yyyy-MM-dd");
    int m = dt.date().month();
    int y = dt.date().year();
    if (m < 1 || m > 12) return month;
    return names[m - 1] + " " + QString::number(y);
}
