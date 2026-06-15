import QtQuick
import QtQuick.Controls

Item {
    id: root
    property string month: budgetManager.currentMonth()
    property int filterType: 0  // 0=all, 1=income, 2=expense

    function startOfMonth(m) { return m + "-01" }
    function endOfMonth(m)   { return m + "-31" }
    function txList() {
        var all = budgetManager.getTransactions(userManager.currentUserId, startOfMonth(month), endOfMonth(month))
        if (filterType === 0) return all
        return all.filter(function(t) { return filterType === 1 ? !t.isExpense : t.isExpense })
    }

    AeroHeader {
        id: hdr
        anchors { top: parent.top; left: parent.left; right: parent.right }
        title: "Операции"
    }

    // Month selector
    Row {
        id: monthRow
        anchors { top: hdr.bottom; topMargin: 12; horizontalCenter: parent.horizontalCenter }
        spacing: 12

        Rectangle {
            width: 36; height: 36; radius: 18
            color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1
            Text { anchors.centerIn: parent; text: "‹"; color: "white"; font.pixelSize: 24 }
            MouseArea { anchors.fill: parent; onClicked: { root.month = budgetManager.prevMonth(root.month); txList() } }
        }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: budgetManager.monthDisplayName(root.month)
            color: "white"; font.pixelSize: 16; font.weight: Font.Medium
        }
        Rectangle {
            width: 36; height: 36; radius: 18
            color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1
            Text { anchors.centerIn: parent; text: "›"; color: "white"; font.pixelSize: 24 }
            MouseArea { anchors.fill: parent; onClicked: { root.month = budgetManager.nextMonth(root.month); txList() } }
        }
    }

    // Filter buttons
    Row {
        id: filterRow
        anchors { top: monthRow.bottom; topMargin: 10; horizontalCenter: parent.horizontalCenter }
        spacing: 8

        Repeater {
            model: ["Все", "Доход", "Расход"]
            delegate: Rectangle {
                width: 80; height: 32; radius: 16
                color: root.filterType === index ? "#1565C0" : Qt.rgba(1,1,1,0.09)
                border.color: Qt.rgba(1,1,1,0.22); border.width: 1
                Behavior on color { ColorAnimation { duration: 150 } }
                Text { anchors.centerIn: parent; text: modelData; color: "white"; font.pixelSize: 13 }
                MouseArea { anchors.fill: parent; onClicked: { root.filterType = index; txListView.model = root.txList() } }
            }
        }
    }

    ListView {
        id: txListView
        anchors { top: filterRow.bottom; topMargin: 12; bottom: parent.bottom; bottomMargin: 8; left: parent.left; leftMargin: 16; right: parent.right; rightMargin: 16 }
        spacing: 10; clip: true
        model: root.txList()

        delegate: TransactionItem {
            width: txListView.width
            tx: modelData
            onDeleteRequested: function(id) {
                budgetManager.deleteTransaction(id)
                txListView.model = root.txList()
            }
        }

        Text {
            visible: txListView.count === 0
            anchors.centerIn: parent
            text: "Операций за этот период нет"
            color: Qt.rgba(1,1,1,0.40); font.pixelSize: 15
        }
    }

    Connections {
        target: budgetManager
        function onDataChanged() { txListView.model = root.txList() }
    }
}
