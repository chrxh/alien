<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $id = $_GET["id"];

    $db->query("UPDATE simulation Set NUM_DOWNLOADS = NUM_DOWNLOADS + 1 where ID=$id");

    if ($response = $db->query("SELECT sim.id as id, sim.content as content FROM simulation sim where ID=$id")) {
        $obj = $response->fetch_object();
        echo $obj->content;
    }
    else {
        echo json_encode(["result"=>false]);
    }

    $db->commit();
    $db->close();
?>