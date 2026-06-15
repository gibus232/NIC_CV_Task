import QtQuick
import QtQuick.Controls
import QtCharts

Item {
    id: root
    property string month: budgetManager.currentMonth()

    function loadData() {
        var cats = budgetManager.getExpensesByCategory(userManager.currentUserId, month)
        pieSeries.clear()
        catList.model = cats
        var total = 0
        for (var i = 0; i < cats.length; i++) total += cats[i].total
        totalExpenses = total
        for (var j = 0; j < cats.length; j++) {
            var s = pieSeries.append(cats[j].name, cats[j].total)
            s.color = cats[j].color || "#1565C0"
            s.borderColor = Qt.rgba(1,1,1,0.15)
        }
        income = budgetManager.getTotalIncome(userManager.currentUserId, month)
    }

    property double totalExpenses: 0
    property double income: 0

    Component.onCompleted: loadData()
    Connections { target: budgetManager; function onDataChanged() { loadData() } }

    AeroHeader {
        id: hdr
        anchors { top: parent.top; left: parent.left; right: parent.right }
        title: "Отчёты"
    }

    Row {
        id: monthRow
        anchors { top: hdr.bottom; topMargin: 12; horizontalCenter: parent.horizontalCenter }; spacing: 12
        Rectangle { width: 36; height: 36; radius: 18; color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1; Text { anchors.centerIn: parent; text: "‹"; color: "white"; font.pixelSize: 24 }; MouseArea { anchors.fill: parent; onClicked: { root.month = budgetManager.prevMonth(root.month); root.loadData() } } }
        Text { anchors.verticalCenter: parent.verticalCenter; text: budgetManager.monthDisplayName(root.month); color: "white"; font.pixelSize: 16; font.weight: Font.Medium }
        Rectangle { width: 36; height: 36; radius: 18; color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1; Text { anchors.centerIn: parent; text: "›"; color: "white"; font.pixelSize: 24 }; MouseArea { anchors.fill: parent; onClicked: { root.month = budgetManager.nextMonth(root.month); root.loadData() } } }
    }

    // Income/Expense summary
    Row {
        id: summaryRow
        anchors { top: monthRow.bottom; topMargin: 10; horizontalCenter: parent.horizontalCenter }; spacing: 12

        Rectangle {
            width: 140; height: 64; radius: 16
            color: Qt.rgba(0.1,0.45,0.1,0.22); border.color: Qt.rgba(0.4,0.9,0.4,0.25); border.width: 1
            Column { anchors.centerIn: parent; spacing: 4
                Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Доход"; color: "#A5D6A7"; font.pixelSize: 12 }
                Text { anchors.horizontalCenter: parent.horizontalCenter; text: Number(root.income).toLocaleString(Qt.locale("ru_RU"),"f",0)+" ₽"; color: "#66BB6A"; font.pixelSize: 17; font.weight: Font.SemiBold }
            }
        }
        Rectangle {
            width: 140; height: 64; radius: 16
            color: Qt.rgba(0.45,0.1,0.1,0.22); border.color: Qt.rgba(0.9,0.35,0.35,0.25); border.width: 1
            Column { anchors.centerIn: parent; spacing: 4
                Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Расход"; color: "#EF9A9A"; font.pixelSize: 12 }
                Text { anchors.horizontalCenter: parent.horizontalCenter; text: Number(root.totalExpenses).toLocaleString(Qt.locale("ru_RU"),"f",0)+" ₽"; color: "#EF5350"; font.pixelSize: 17; font.weight: Font.SemiBold }
            }
        }
    }

    // Pie chart
    ChartView {
        id: chart
        anchors { top: summaryRow.bottom; topMargin: 4; left: parent.left; right: parent.right }
        height: 220
        backgroundColor: "transparent"
        legend.visible: false
        antialiasing: true
        animationOptions: ChartView.AllAnimations

        PieSeries {
            id: pieSeries
            holeSize: 0.48
            size: 0.88
        }
    }

    // Category breakdown
    ListView {
        id: catList
        anchors { top: chart.bottom; bottom: parent.bottom; bottomMargin: 8; left: parent.left; leftMargin: 16; right: parent.right; rightMargin: 16 }
        spacing: 8; clip: true

        delegate: Rectangle {
            width: catList.width; height: 48; radius: 12
            color: Qt.rgba(1,1,1,0.07); border.color: Qt.rgba(1,1,1,0.14); border.width: 1

            Row {
                anchors { fill: parent; leftMargin: 12; rightMargin: 12 }; spacing: 10
                Rectangle { width: 8; height: 8; radius: 4; anchors.verticalCenter: parent.verticalCenter; color: modelData.color || "#1565C0" }
                Text { anchors.verticalCenter: parent.verticalCenter; text: modelData.name; color: "white"; font.pixelSize: 14; width: parent.width - 130; elide: Text.ElideRight }
                Text { anchors.verticalCenter: parent.verticalCenter; text: Number(modelData.total).toLocaleString(Qt.locale("ru_RU"),"f",0)+" ₽"; color: Qt.rgba(1,1,1,0.75); font.pixelSize: 14; width: 100; horizontalAlignment: Text.AlignRight }
            }
        }

        Text { visible: catList.count === 0; anchors.centerIn: parent; text: "Расходов за этот месяц нет"; color: Qt.rgba(1,1,1,0.40); font.pixelSize: 15 }
    }
}
