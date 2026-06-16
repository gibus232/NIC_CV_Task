#include "database.h"
#include <QStandardPaths>
#include <QDir>
#include <QDebug>
#include <QCryptographicHash>

Database::Database(QObject *parent) : QObject(parent) {}

Database::~Database()
{
    if (m_db.isOpen())
        m_db.close();
}

bool Database::initialize()
{
    QString path = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(path);
    path += "/familybudget.db";

    m_db = QSqlDatabase::addDatabase("QSQLITE");
    m_db.setDatabaseName(path);

    if (!m_db.open()) {
        qCritical() << "DB open failed:" << m_db.lastError().text();
        return false;
    }
    return createTables();
}

bool Database::createTables()
{
    QSqlQuery q(m_db);
    q.exec("PRAGMA foreign_keys = ON");
    q.exec("PRAGMA journal_mode = WAL");

    auto exec = [&](const char *sql) {
        if (!q.exec(sql)) {
            qCritical() << q.lastError().text();
            return false;
        }
        return true;
    };

    if (!exec(R"(
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            pin_hash   TEXT,
            color      TEXT    DEFAULT '#1E90FF',
            avatar     TEXT    DEFAULT 'person',
            created_at TEXT    DEFAULT (datetime('now'))
        )
    )")) return false;

    if (!exec(R"(
        CREATE TABLE IF NOT EXISTS categories (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            icon       TEXT    DEFAULT 'category',
            color      TEXT    DEFAULT '#1E90FF',
            is_expense INTEGER DEFAULT 1,
            user_id    INTEGER DEFAULT 0
        )
    )")) return false;

    if (!exec(R"(
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            amount      REAL    NOT NULL,
            is_expense  INTEGER DEFAULT 1,
            note        TEXT    DEFAULT '',
            date        TEXT    NOT NULL,
            created_at  TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY(user_id)     REFERENCES users(id)      ON DELETE CASCADE,
            FOREIGN KEY(category_id) REFERENCES categories(id)
        )
    )")) return false;

    if (!exec(R"(
        CREATE TABLE IF NOT EXISTS budgets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            amount      REAL    NOT NULL,
            month       TEXT    NOT NULL,
            UNIQUE(user_id, category_id, month),
            FOREIGN KEY(user_id)     REFERENCES users(id)      ON DELETE CASCADE,
            FOREIGN KEY(category_id) REFERENCES categories(id)
        )
    )")) return false;

    QSqlQuery check(m_db);
    check.exec("SELECT COUNT(*) FROM categories WHERE user_id = 0");
    if (check.next() && check.value(0).toInt() == 0)
        seedDefaultCategories();

    return true;
}

void Database::seedDefaultCategories()
{
    struct Cat { const char *name, *icon, *color; int expense; };
    static const Cat cats[] = {
        // expenses
        {"\xD0\x95\xD0\xB4\xD0\xB0 \xD0\xB8 \xD1\x80\xD0\xB5\xD1\x81\xD1\x82\xD0\xBE\xD1\x80\xD0\xB0\xD0\xBD\xD1\x8B", "restaurant",   "#FF6B35", 1},
        {"\xD0\xA2\xD1\x80\xD0\xB0\xD0\xBD\xD1\x81\xD0\xBF\xD0\xBE\xD1\x80\xD1\x82",              "directions_car","#4ECDC4", 1},
        {"\xD0\x96\xD0\xB8\xD0\xBB\xD1\x8C\xD1\x91",                               "home",          "#45B7D1", 1},
        {"\xD0\x97\xD0\xB4\xD0\xBE\xD1\x80\xD0\xBE\xD0\xB2\xD1\x8C\xD0\xB5",     "local_hospital","#96CEB4", 1},
        {"\xD0\xA0\xD0\xB0\xD0\xB7\xD0\xB2\xD0\xBB\xD0\xB5\xD1\x87\xD0\xB5\xD0\xBD\xD0\xB8\xD1\x8F","theaters",      "#FFEAA7", 1},
        {"\xD0\x9E\xD0\xB4\xD0\xB5\xD0\xB6\xD0\xB4\xD0\xB0",                      "checkroom",     "#DDA0DD", 1},
        {"\xD0\x9E\xD0\xB1\xD1\x80\xD0\xB0\xD0\xB7\xD0\xBE\xD0\xB2\xD0\xB0\xD0\xBD\xD0\xB8\xD0\xB5","school",        "#98D8C8", 1},
        {"\xD0\xA1\xD0\xB2\xD1\x8F\xD0\xB7\xD1\x8C",                               "phone_android", "#AED6F1", 1},
        {"\xD0\xA1\xD0\xBF\xD0\xBE\xD1\x80\xD1\x82",                               "fitness_center","#A8E6CF", 1},
        {"\xD0\x9F\xD1\x80\xD0\xBE\xD1\x87\xD0\xB5\xD0\xB5",                      "more_horiz",    "#B0BEC5", 1},
        // income
        {"\xD0\x97\xD0\xB0\xD1\x80\xD0\xBF\xD0\xBB\xD0\xB0\xD1\x82\xD0\xB0",    "work",          "#26A65B", 0},
        {"\xD0\xA4\xD1\x80\xD0\xB8\xD0\xBB\xD0\xB0\xD0\xBD\xD1\x81",             "laptop",        "#1E8BC3", 0},
        {"\xD0\x98\xD0\xBD\xD0\xB2\xD0\xB5\xD1\x81\xD1\x82\xD0\xB8\xD1\x86\xD0\xB8\xD0\xB8","trending_up",   "#2ECC71", 0},
        {"\xD0\x9F\xD0\xBE\xD0\xB4\xD0\xB0\xD1\x80\xD0\xBA\xD0\xB8",             "card_giftcard", "#9B59B6", 0},
        {"\xD0\x9F\xD1\x80\xD0\xBE\xD1\x87\xD0\xB5\xD0\xB5",                      "more_horiz",    "#7F8C8D", 0},
    };

    QSqlQuery q(m_db);
    q.prepare("INSERT INTO categories (name, icon, color, is_expense, user_id) VALUES (?,?,?,?,0)");
    for (auto &c : cats) {
        q.bindValue(0, QString::fromUtf8(c.name));
        q.bindValue(1, c.icon);
        q.bindValue(2, c.color);
        q.bindValue(3, c.expense);
        q.exec();
    }
}

static QString hashPin(const QString &pin)
{
    return QCryptographicHash::hash(pin.toUtf8(), QCryptographicHash::Sha256).toHex();
}

int Database::createUser(const QString &name, const QString &pin,
                          const QString &color, const QString &avatar)
{
    QSqlQuery q(m_db);
    q.prepare("INSERT INTO users (name, pin_hash, color, avatar) VALUES (?,?,?,?)");
    q.bindValue(0, name);
    q.bindValue(1, pin.isEmpty() ? QString() : hashPin(pin));
    q.bindValue(2, color);
    q.bindValue(3, avatar);
    return q.exec() ? q.lastInsertId().toInt() : -1;
}

bool Database::updateUser(int id, const QString &name, const QString &pin,
                           const QString &color, const QString &avatar)
{
    QSqlQuery q(m_db);
    if (pin.isEmpty()) {
        q.prepare("UPDATE users SET name=?, color=?, avatar=? WHERE id=?");
        q.bindValue(0, name); q.bindValue(1, color);
        q.bindValue(2, avatar); q.bindValue(3, id);
    } else {
        q.prepare("UPDATE users SET name=?, pin_hash=?, color=?, avatar=? WHERE id=?");
        q.bindValue(0, name); q.bindValue(1, hashPin(pin));
        q.bindValue(2, color); q.bindValue(3, avatar); q.bindValue(4, id);
    }
    return q.exec();
}

bool Database::deleteUser(int id)
{
    QSqlQuery q(m_db);
    q.prepare("DELETE FROM users WHERE id=?");
    q.bindValue(0, id);
    return q.exec();
}

QList<QVariantMap> Database::getAllUsers()
{
    QSqlQuery q(m_db);
    q.exec("SELECT id, name, color, avatar FROM users ORDER BY id");
    QList<QVariantMap> r;
    while (q.next()) {
        r << QVariantMap{{">id",q.value(0)},{"id",q.value(0)},{"name",q.value(1)},
                         {"color",q.value(2)},{"avatar",q.value(3)}};
    }
    return r;
}

QVariantMap Database::getUserById(int id)
{
    QSqlQuery q(m_db);
    q.prepare("SELECT id, name, color, avatar FROM users WHERE id=?");
    q.bindValue(0, id);
    q.exec();
    if (q.next())
        return {{"id",q.value(0)},{"name",q.value(1)},
                {"color",q.value(2)},{"avatar",q.value(3)}};
    return {};
}

bool Database::verifyPin(int userId, const QString &pin)
{
    QSqlQuery q(m_db);
    q.prepare("SELECT pin_hash FROM users WHERE id=?");
    q.bindValue(0, userId); q.exec();
    if (q.next()) {
        QString stored = q.value(0).toString();
        if (stored.isEmpty()) return true;
        return stored == hashPin(pin);
    }
    return false;
}

bool Database::userHasPin(int userId)
{
    QSqlQuery q(m_db);
    q.prepare("SELECT pin_hash FROM users WHERE id=?");
    q.bindValue(0, userId); q.exec();
    return q.next() && !q.value(0).toString().isEmpty();
}

int Database::createCategory(const QString &name, const QString &icon,
                              const QString &color, bool isExpense, int userId)
{
    QSqlQuery q(m_db);
    q.prepare("INSERT INTO categories (name, icon, color, is_expense, user_id) VALUES (?,?,?,?,?)");
    q.bindValue(0, name); q.bindValue(1, icon); q.bindValue(2, color);
    q.bindValue(3, isExpense ? 1 : 0); q.bindValue(4, userId);
    return q.exec() ? q.lastInsertId().toInt() : -1;
}

bool Database::updateCategory(int id, const QString &name,
                               const QString &icon, const QString &color)
{
    QSqlQuery q(m_db);
    q.prepare("UPDATE categories SET name=?, icon=?, color=? WHERE id=?");
    q.bindValue(0, name); q.bindValue(1, icon);
    q.bindValue(2, color); q.bindValue(3, id);
    return q.exec();
}

bool Database::deleteCategory(int id)
{
    QSqlQuery q(m_db);
    q.prepare("DELETE FROM categories WHERE id=?");
    q.bindValue(0, id);
    return q.exec();
}

QList<QVariantMap> Database::getCategories(int userId, bool isExpense)
{
    QSqlQuery q(m_db);
    q.prepare("SELECT id, name, icon, color FROM categories WHERE (user_id=0 OR user_id=?) AND is_expense=? ORDER BY name");
    q.bindValue(0, userId); q.bindValue(1, isExpense ? 1 : 0);
    q.exec();
    QList<QVariantMap> r;
    while (q.next())
        r << QVariantMap{{"id",q.value(0)},{"name",q.value(1)},
                         {"icon",q.value(2)},{"color",q.value(3)}};
    return r;
}

QList<QVariantMap> Database::getAllCategories(int userId)
{
    QSqlQuery q(m_db);
    q.prepare("SELECT id, name, icon, color, is_expense FROM categories WHERE user_id=0 OR user_id=? ORDER BY is_expense DESC, name");
    q.bindValue(0, userId); q.exec();
    QList<QVariantMap> r;
    while (q.next())
        r << QVariantMap{{"id",q.value(0)},{"name",q.value(1)},{"icon",q.value(2)},
                         {"color",q.value(3)},{"isExpense",q.value(4).toBool()}};
    return r;
}

int Database::addTransaction(int userId, int categoryId, double amount,
                              bool isExpense, const QString &note, const QString &date)
{
    QSqlQuery q(m_db);
    q.prepare("INSERT INTO transactions (user_id, category_id, amount, is_expense, note, date) VALUES (?,?,?,?,?,?)");
    q.bindValue(0, userId); q.bindValue(1, categoryId); q.bindValue(2, amount);
    q.bindValue(3, isExpense ? 1 : 0); q.bindValue(4, note); q.bindValue(5, date);
    return q.exec() ? q.lastInsertId().toInt() : -1;
}

bool Database::updateTransaction(int id, int categoryId, double amount,
                                  bool isExpense, const QString &note, const QString &date)
{
    QSqlQuery q(m_db);
    q.prepare("UPDATE transactions SET category_id=?, amount=?, is_expense=?, note=?, date=? WHERE id=?");
    q.bindValue(0, categoryId); q.bindValue(1, amount); q.bindValue(2, isExpense ? 1 : 0);
    q.bindValue(3, note); q.bindValue(4, date); q.bindValue(5, id);
    return q.exec();
}

bool Database::deleteTransaction(int id)
{
    QSqlQuery q(m_db);
    q.prepare("DELETE FROM transactions WHERE id=?");
    q.bindValue(0, id);
    return q.exec();
}

QList<QVariantMap> Database::getTransactions(int userId,
                                              const QString &startDate, const QString &endDate)
{
    QSqlQuery q(m_db);
    q.prepare(R"(
        SELECT t.id, t.amount, t.is_expense, t.note, t.date,
               c.name, c.icon, c.color
        FROM transactions t JOIN categories c ON t.category_id = c.id
        WHERE t.user_id=? AND t.date >= ? AND t.date <= ?
        ORDER BY t.date DESC, t.id DESC
    )");
    q.bindValue(0, userId); q.bindValue(1, startDate); q.bindValue(2, endDate);
    q.exec();
    QList<QVariantMap> r;
    while (q.next())
        r << QVariantMap{{"id",q.value(0)},{"amount",q.value(1)},
                         {"isExpense",q.value(2).toBool()},{"note",q.value(3)},
                         {"date",q.value(4)},{"categoryName",q.value(5)},
                         {"categoryIcon",q.value(6)},{"categoryColor",q.value(7)}};
    return r;
}

QList<QVariantMap> Database::getRecentTransactions(int userId, int limit)
{
    QSqlQuery q(m_db);
    q.prepare(R"(
        SELECT t.id, t.amount, t.is_expense, t.note, t.date,
               c.name, c.icon, c.color
        FROM transactions t JOIN categories c ON t.category_id = c.id
        WHERE t.user_id=?
        ORDER BY t.date DESC, t.id DESC LIMIT ?
    )");
    q.bindValue(0, userId); q.bindValue(1, limit); q.exec();
    QList<QVariantMap> r;
    while (q.next())
        r << QVariantMap{{"id",q.value(0)},{"amount",q.value(1)},
                         {"isExpense",q.value(2).toBool()},{"note",q.value(3)},
                         {"date",q.value(4)},{"categoryName",q.value(5)},
                         {"categoryIcon",q.value(6)},{"categoryColor",q.value(7)}};
    return r;
}

double Database::getTotalIncome(int userId, const QString &month)
{
    QSqlQuery q(m_db);
    q.prepare("SELECT COALESCE(SUM(amount),0) FROM transactions WHERE user_id=? AND is_expense=0 AND strftime('%Y-%m',date)=?");
    q.bindValue(0, userId); q.bindValue(1, month); q.exec();
    return q.next() ? q.value(0).toDouble() : 0.0;
}

double Database::getTotalExpenses(int userId, const QString &month)
{
    QSqlQuery q(m_db);
    q.prepare("SELECT COALESCE(SUM(amount),0) FROM transactions WHERE user_id=? AND is_expense=1 AND strftime('%Y-%m',date)=?");
    q.bindValue(0, userId); q.bindValue(1, month); q.exec();
    return q.next() ? q.value(0).toDouble() : 0.0;
}

QList<QVariantMap> Database::getExpensesByCategory(int userId, const QString &month)
{
    QSqlQuery q(m_db);
    q.prepare(R"(
        SELECT c.name, c.color, c.icon, COALESCE(SUM(t.amount),0) as total
        FROM categories c
        LEFT JOIN transactions t ON c.id=t.category_id
            AND t.user_id=? AND t.is_expense=1 AND strftime('%Y-%m',t.date)=?
        WHERE (c.user_id=0 OR c.user_id=?) AND c.is_expense=1
        GROUP BY c.id HAVING total > 0 ORDER BY total DESC
    )");
    q.bindValue(0, userId); q.bindValue(1, month); q.bindValue(2, userId); q.exec();
    QList<QVariantMap> r;
    while (q.next())
        r << QVariantMap{{"name",q.value(0)},{"color",q.value(1)},
                         {"icon",q.value(2)},{"total",q.value(3)}};
    return r;
}

QList<QVariantMap> Database::getMonthlyTrend(int userId, int months)
{
    QSqlQuery q(m_db);
    q.prepare(R"(
        SELECT strftime('%Y-%m', date) as month,
               SUM(CASE WHEN is_expense=0 THEN amount ELSE 0 END) as income,
               SUM(CASE WHEN is_expense=1 THEN amount ELSE 0 END) as expenses
        FROM transactions WHERE user_id=?
        GROUP BY month ORDER BY month DESC LIMIT ?
    )");
    q.bindValue(0, userId); q.bindValue(1, months); q.exec();
    QList<QVariantMap> r;
    while (q.next())
        r << QVariantMap{{"month",q.value(0)},{"income",q.value(1)},
                         {"expenses",q.value(2)}};
    return r;
}

bool Database::setBudget(int userId, int categoryId, double amount, const QString &month)
{
    QSqlQuery q(m_db);
    q.prepare("INSERT OR REPLACE INTO budgets (user_id, category_id, amount, month) VALUES (?,?,?,?)");
    q.bindValue(0, userId); q.bindValue(1, categoryId);
    q.bindValue(2, amount); q.bindValue(3, month);
    return q.exec();
}

double Database::getBudget(int userId, int categoryId, const QString &month)
{
    QSqlQuery q(m_db);
    q.prepare("SELECT amount FROM budgets WHERE user_id=? AND category_id=? AND month=?");
    q.bindValue(0, userId); q.bindValue(1, categoryId); q.bindValue(2, month); q.exec();
    return q.next() ? q.value(0).toDouble() : 0.0;
}

QList<QVariantMap> Database::getBudgets(int userId, const QString &month)
{
    QSqlQuery q(m_db);
    q.prepare(R"(
        SELECT b.category_id, c.name, c.color, c.icon, b.amount,
               COALESCE(SUM(t.amount),0) as spent
        FROM budgets b JOIN categories c ON b.category_id=c.id
        LEFT JOIN transactions t ON t.category_id=b.category_id
            AND t.user_id=b.user_id AND t.is_expense=1
            AND strftime('%Y-%m',t.date)=b.month
        WHERE b.user_id=? AND b.month=?
        GROUP BY b.category_id ORDER BY c.name
    )");
    q.bindValue(0, userId); q.bindValue(1, month); q.exec();
    QList<QVariantMap> r;
    while (q.next())
        r << QVariantMap{{"categoryId",q.value(0)},{"name",q.value(1)},{"color",q.value(2)},
                         {"icon",q.value(3)},{"budget",q.value(4)},{"spent",q.value(5)}};
    return r;
}
