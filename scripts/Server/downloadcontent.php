<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $id = $_GET["id"];
    $chunkIndex = array_key_exists("chunkIndex", $_GET) ? (int)$_GET["chunkIndex"] : 0;

    if ($chunkIndex == 0) {
        $db->query("UPDATE simulation Set NUM_DOWNLOADS = NUM_DOWNLOADS + 1, TIMESTAMP = TIMESTAMP where ID=$id");
    }
    if ($chunkIndex > 5) {
        echo "";
        $db->commit();
        $db->close();
    }

    $chunkString = $chunkIndex == 0 ? "" : (string)($chunkIndex + 1);
    if ($response = $db->query("SELECT sim.id as id, sim.content" . $chunkString . " as content FROM simulation sim where ID=$id")) {
        $obj = $response->fetch_object();
        echo $obj->content;
    }
    else {
        echo json_encode(["result"=>false]);
    }

    $db->commit();
    $db->close();
?>