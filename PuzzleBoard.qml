import QtQuick 2.15

GridView {
    id: puzzleGrid
    // anchors.centerIn: parent
    width: Math.min(parent.width, parent.height)
    height: width

    interactive: false

    property alias model: puzzleGrid.model
    property alias tileSize: puzzleGrid.cellWidth
    property int gridSize: Math.sqrt(model.count)

    cellWidth: width / gridSize
    cellHeight: cellWidth

    delegate: Rectangle {
        width: puzzleGrid.cellWidth
        height: puzzleGrid.cellHeight
        color: modelData === 0 ? "lightgray" : "white"
        border.color: "black"

        Text {
            anchors.centerIn: parent
            text: modelData === 0 ? "" : modelData
            font.pixelSize: 24
        }
    }
}
